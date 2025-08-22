import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import asyncio
import chromadb
from chromadb.config import Settings
import structlog
import voyageai
from pydantic import BaseModel, SecretStr
from dotenv import load_dotenv
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger(__name__)

# Set protobuf compatibility
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


class VectorStoreConfig(BaseModel):
    voyage_api_key: SecretStr
    model_name: str = "voyage-3-large"
    embedding_dimension: int = 1536
    persist_directory: str = "./chroma_db"
    batch_size: int = 100
    top_k: int = 5


class VectorStoreService:
    """Manages conversation data using ChromaDB and VoyageAI embeddings."""

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """Initialize with configuration."""
        self.config = config or VectorStoreConfig(
            voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        )
        try:
            self.voyage_client = voyageai.Client(api_key=self.config.voyage_api_key.get_secret_value())
            os.makedirs(self.config.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=self.config.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info(f"Initialized vector store - model: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _get_collection_name(self, userId: str) -> str:
        """Generate collection name for a user."""
        userId = userId or "default"
        return f"user_{userId.replace('-', '_').replace(' ', '_').replace('@', '_at_')}"

    def _get_collection(self, userId: str) -> Optional[chromadb.Collection]:
        """Retrieve or create user collection."""
        try:
            return self.client.get_or_create_collection(self._get_collection_name(userId))
        except Exception as e:
            logger.error(f"Failed to get collection - userId: {userId}, error: {str(e)}")
            return None

    def _format_where_filter(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Format filter dictionary for ChromaDB query."""
        if not filter_dict:
            return {}
        str_filter = {k: str(v) for k, v in filter_dict.items() if v is not None}
        return (
            {"$and": [{k: {"$eq": v}} for k, v in str_filter.items()]}
            if len(str_filter) > 1
            else {k: {"$eq": v} for k, v in str_filter.items()}
        )

    async def add_chat_messages(
            self,
            userId: str,
            messages: List[Dict[str, Any]],
            knowledgebaseId: Optional[str] = None,
            model_name: Optional[str] = None
    ) -> bool:
        """Add chat messages to vector store in batches."""
        logger.info(f"VECTOR STORE - Adding {len(messages) if messages else 0} messages to vector store", 
                   userId=userId, knowledgebaseId=knowledgebaseId, model_name=model_name)
        
        if not messages or not self.client or not self.voyage_client:
            logger.warning("VECTOR STORE - Invalid input or clients not initialized")
            return False

        collection = self._get_collection(userId)
        if not collection:
            logger.error(f"VECTOR STORE - Failed to get collection for user: {userId}")
            return False

        documents = []
        metadatas = []
        ids = []

        for msg in messages:
            if not msg.get("body"):
                continue
            doc_id = str(msg.get("id", uuid.uuid4()))
            ids.append(doc_id)
            documents.append(msg["body"])
            now = datetime.now(timezone.utc).isoformat()
            metadata = {
                "userId": str(userId),
                "id": doc_id,
                "conversationId": str(msg.get("conversationId", "")),
                "knowledgebaseId": str(knowledgebaseId or msg.get("conversationId", "")),
                "direction": str(msg.get("direction", "")),
                "role": str(msg.get("role", "")),
                "messageType": str(msg.get("messageType", "text")),
                "contentType": str(msg.get("contentType", "text/plain")),
                "source": str(msg.get("source", "conversation")),
                "dateAdded": str(msg.get("dateAdded", now)),
                "locationId": str(msg.get("locationId", "default")),
            }
            metadatas.append(metadata)

        if not documents:
            logger.warning("VECTOR STORE - No valid messages to add after processing")
            return False

        logger.info(f"VECTOR STORE - Processing {len(documents)} documents for embedding")
        model = model_name or self.config.model_name
        logger.debug(f"VECTOR STORE - Using embedding model: {model}")
        
        try:
            total_batches = (len(documents) + self.config.batch_size - 1) // self.config.batch_size
            for batch_idx, i in enumerate(range(0, len(documents), self.config.batch_size)):
                batch_texts = documents[i:i + self.config.batch_size]
                batch_metadatas = metadatas[i:i + self.config.batch_size]
                batch_ids = ids[i:i + self.config.batch_size]

                logger.info(f"VECTOR STORE - Processing batch {batch_idx + 1}/{total_batches} with {len(batch_texts)} documents")
                
                response = await asyncio.to_thread(
                    self.voyage_client.embed,
                    texts=batch_texts,
                    model=model,
                    input_type="document",
                    truncation=True
                )
                if not response.embeddings:
                    logger.error("VECTOR STORE - Failed to generate embeddings")
                    return False

                logger.debug(f"VECTOR STORE - Generated {len(response.embeddings)} embeddings")
                
                collection.add(
                    embeddings=response.embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                logger.info(f"VECTOR STORE - Added batch {batch_idx + 1} to collection - count: {len(batch_texts)}")
            
            logger.info(f"VECTOR STORE - Successfully added all {len(documents)} documents to vector store")
            return True
        except Exception as e:
            logger.error(f"Failed to add messages: {str(e)}")
            return False

    async def get_conversation_by_id(
            self,
            userId: str,
            conversationId: str,
            knowledgebaseId: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve all messages for a conversation ID."""
        collection = self._get_collection(userId)
        if not collection:
            return []

        metadata_filter = {"conversationId": conversationId}
        if knowledgebaseId:
            metadata_filter["knowledgebaseId"] = knowledgebaseId

        try:
            results = collection.get(where=self._format_where_filter(metadata_filter))
            return [
                {"id": results['ids'][i], "body": results['documents'][i], **results['metadatas'][i]}
                for i in range(len(results['documents']))
            ] if results and results['documents'] else []
        except Exception as e:
            logger.error(f"Failed to retrieve conversation - conversationId: {conversationId}, error: {str(e)}")
            return []

    async def query_chat_history(
            self,
            userId: str,
            query: str,
            metadata_filter: Optional[Dict[str, Any]] = None,
            k: Optional[int] = None,
            knowledgebaseId: Optional[str] = None,
            model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query chat history with optional metadata filter."""
        logger.info(f"VECTOR STORE - Querying chat history - userId: {userId}, query_length: {len(query)}, k: {k}, knowledgebaseId: {knowledgebaseId}")
        
        collection = self._get_collection(userId)
        if not collection or not self.voyage_client:
            logger.warning(f"VECTOR STORE - Collection or voyage client not available - userId: {userId}")
            return []

        metadata_filter = metadata_filter or {}
        if knowledgebaseId:
            metadata_filter["knowledgebaseId"] = knowledgebaseId

        formatted_filter = self._format_where_filter(metadata_filter)
        model = model_name or self.config.model_name
        k = k or self.config.top_k
        
        # Debug logging
        logger.debug(f"Query chat history debug - userId: {userId}, query_length: {len(query) if query else 0}, k: {k}, model: {model}")

        if not query or not query.strip():
            try:
                results = collection.get(where=formatted_filter, limit=k)
                result_count = len(results['documents']) if results and results['documents'] else 0
                logger.info(f"Metadata-only search completed - result_count: {result_count}")
                return [
                    {"id": results['ids'][i], "body": results['documents'][i], "score": 1.0, **results['metadatas'][i]}
                    for i in range(len(results['documents']))
                ] if results and results['documents'] else []
            except Exception as e:
                logger.error(f"Metadata search failed: {str(e)}")
                return []

        try:
            response = await asyncio.to_thread(
                self.voyage_client.embed,
                texts=[query],
                model=model,
                input_type="query",
                truncation=True
            )
            
            if not response.embeddings:
                logger.error("Failed to generate query embeddings")
                return []
                
            results = collection.query(
                query_embeddings=response.embeddings,
                where=formatted_filter,
                n_results=k
            )
            
            result_count = len(results['documents'][0]) if results and results['documents'] and results['documents'][0] else 0
            logger.info(f"Semantic search completed - result_count: {result_count}")
            
            return [
                {
                    "id": results['ids'][0][i],
                    "body": results['documents'][0][i],
                    "score": float(results['distances'][0][i]),
                    **results['metadatas'][0][i]
                }
                for i in range(len(results['documents'][0]))
            ] if results and results['documents'] else []
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return []

    async def get_message_count(
            self,
            userId: str,
            metadata_filter: Optional[Dict[str, Any]] = None,
            knowledgebaseId: Optional[str] = None
    ) -> int:
        """Count messages matching the filter."""
        collection = self._get_collection(userId)
        if not collection:
            return 0

        metadata_filter = metadata_filter or {}
        if knowledgebaseId:
            metadata_filter["knowledgebaseId"] = knowledgebaseId

        try:
            results = collection.get(where=self._format_where_filter(metadata_filter))
            return len(results['documents']) if results and results['documents'] else 0
        except Exception as e:
            logger.error(f"Failed to count messages: {str(e)}")
            return 0

    async def delete_user_data(self, userId: str) -> bool:
        """Delete all user data."""
        try:
            self.client.delete_collection(self._get_collection_name(userId))
            logger.info(f"Deleted user data - userId: {userId}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete user data: {str(e)}")
            return False

    async def delete_conversation(
            self,
            userId: str,
            conversationId: str,
            knowledgebaseId: Optional[str] = None
    ) -> bool:
        """Delete all messages for a specific conversation."""
        collection = self._get_collection(userId)
        if not collection:
            return False

        metadata_filter = {"conversationId": conversationId}
        if knowledgebaseId:
            metadata_filter["knowledgebaseId"] = knowledgebaseId

        try:
            results = collection.get(where=self._format_where_filter(metadata_filter))
            if results and results['ids']:
                collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} messages for conversation {conversationId}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete conversation - conversationId: {conversationId}, error: {str(e)}")
            return False

    async def add_documents(
            self,
            documents: List[Document],
            knowledgebaseId: Optional[str] = None,
            model_name: Optional[str] = None
    ) -> bool:
        """Add Document objects to vector store by converting them to message format."""
        if not documents:
            logger.warning("VECTOR STORE - No documents provided")
            return False
        
        # Convert Document objects to message format
        messages = []
        for doc in documents:
            # Extract metadata from Document object
            metadata = getattr(doc, 'metadata', {})
            content = getattr(doc, 'page_content', '') or getattr(doc, 'content', '')
            
            # Convert to message format expected by add_chat_messages
            message_data = {
                "id": metadata.get("messageId", str(uuid.uuid4())),
                "body": content,
                "direction": metadata.get("direction", "inbound"),
                "role": metadata.get("role", "web_content"),
                "conversationId": metadata.get("conversationId", ""),
                "knowledgebaseId": metadata.get("knowledgebaseId", knowledgebaseId),
                "userId": metadata.get("userId", ""),
                "source": metadata.get("source", "document"),
                "contentType": metadata.get("contentType", "text/plain"),
                "messageType": metadata.get("messageType", "text"),
                "dateAdded": metadata.get("dateAdded", datetime.now(timezone.utc).isoformat()),
                "locationId": metadata.get("locationId", "default"),
                "type": metadata.get("type", "document")
            }
            messages.append(message_data)
        
        # Use existing add_chat_messages method
        if not messages:
            logger.warning("VECTOR STORE - No valid messages after document conversion")
            return False
            
        # Get userId from first message
        userId = messages[0].get("userId", "default")
        
        logger.info(f"VECTOR STORE - Converting {len(documents)} documents to messages for userId: {userId}")
        
        return await self.add_chat_messages(
            userId=userId,
            messages=messages,
            knowledgebaseId=knowledgebaseId,
            model_name=model_name
        )

    # ============================================================================
    # UPDATE METHODS
    # ============================================================================

    async def update_message(
            self,
            userId: str,
            message_id: str,
            updates: Dict[str, Any],
            knowledgebaseId: Optional[str] = None,
            model_name: Optional[str] = None
    ) -> bool:
        """Update a specific message in the vector store."""
        logger.info(f"VECTOR STORE - Updating message {message_id} for user {userId}")
        
        collection = self._get_collection(userId)
        if not collection:
            logger.error(f"VECTOR STORE - Failed to get collection for user: {userId}")
            return False

        try:
            # Get the existing message
            existing = collection.get(ids=[message_id])
            if not existing or not existing['documents'] or not existing['documents'][0]:
                logger.warning(f"VECTOR STORE - Message {message_id} not found for update")
                return False

            # Extract current data
            current_doc = existing['documents'][0]
            current_metadata = existing['metadatas'][0] if existing['metadatas'] else {}
            
            # Prepare updated content and metadata
            updated_content = updates.get('body', current_doc)
            updated_metadata = current_metadata.copy()
            
            # Update metadata fields
            for field in ['role', 'messageType', 'contentType', 'source', 'status']:
                if field in updates and updates[field] is not None:
                    updated_metadata[field] = str(updates[field])
            
            # Handle additional metadata updates
            if 'metadata' in updates and updates['metadata']:
                for key, value in updates['metadata'].items():
                    updated_metadata[key] = str(value)
            
            # Add update timestamp
            updated_metadata['lastUpdated'] = datetime.now(timezone.utc).isoformat()
            
            # If content changed, regenerate embeddings
            if updated_content != current_doc:
                logger.info(f"VECTOR STORE - Content changed for message {message_id}, regenerating embeddings")
                
                model = model_name or self.config.model_name
                response = await asyncio.to_thread(
                    self.voyage_client.embed,
                    texts=[updated_content],
                    model=model,
                    input_type="document",
                    truncation=True
                )
                
                if not response.embeddings:
                    logger.error("VECTOR STORE - Failed to generate embeddings for update")
                    return False
                
                # Delete the old entry and add the updated one
                collection.delete(ids=[message_id])
                collection.add(
                    embeddings=response.embeddings,
                    documents=[updated_content],
                    metadatas=[updated_metadata],
                    ids=[message_id]
                )
            else:
                # Content unchanged, just update metadata
                collection.update(
                    ids=[message_id],
                    metadatas=[updated_metadata]
                )
            
            logger.info(f"VECTOR STORE - Successfully updated message {message_id}")
            return True
            
        except Exception as e:
            logger.error(f"VECTOR STORE - Failed to update message {message_id}: {str(e)}")
            return False

    async def update_messages_bulk(
            self,
            userId: str,
            updates: List[Dict[str, Any]],
            knowledgebaseId: Optional[str] = None,
            model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update multiple messages in bulk."""
        logger.info(f"VECTOR STORE - Bulk updating {len(updates)} messages for user {userId}")
        
        results = {
            "updated": [],
            "failed": [],
            "errors": []
        }
        
        for update in updates:
            message_id = update.get('id')
            if not message_id:
                results["failed"].append("Unknown")
                results["errors"].append("Missing message ID")
                continue
                
            success = await self.update_message(
                userId=userId,
                message_id=message_id,
                updates=update,
                knowledgebaseId=knowledgebaseId,
                model_name=model_name
            )
            
            if success:
                results["updated"].append(message_id)
            else:
                results["failed"].append(message_id)
                results["errors"].append(f"Failed to update message {message_id}")
        
        logger.info(f"VECTOR STORE - Bulk update completed: {len(results['updated'])} updated, {len(results['failed'])} failed")
        return results

    async def replace_conversation_content(
            self,
            userId: str,
            conversationId: str,
            new_messages: List[Dict[str, Any]],
            knowledgebaseId: Optional[str] = None,
            model_name: Optional[str] = None
    ) -> bool:
        """Replace all content in a conversation with new messages."""
        logger.info(f"VECTOR STORE - Replacing conversation {conversationId} content for user {userId}")
        
        try:
            # Delete existing conversation
            delete_success = await self.delete_conversation(
                userId=userId,
                conversationId=conversationId,
                knowledgebaseId=knowledgebaseId
            )
            
            if not delete_success:
                logger.warning(f"VECTOR STORE - Failed to delete existing conversation {conversationId}")
                return False
            
            # Add new messages
            add_success = await self.add_chat_messages(
                userId=userId,
                messages=new_messages,
                knowledgebaseId=knowledgebaseId or conversationId,
                model_name=model_name
            )
            
            if add_success:
                logger.info(f"VECTOR STORE - Successfully replaced conversation {conversationId} content")
            else:
                logger.error(f"VECTOR STORE - Failed to add new messages to conversation {conversationId}")
            
            return add_success
            
        except Exception as e:
            logger.error(f"VECTOR STORE - Failed to replace conversation content: {str(e)}")
            return False

    async def update_knowledge_base_content(
            self,
            userId: str,
            knowledgebaseId: str,
            updates: List[Dict[str, Any]],
            update_type: str = "partial",
            model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update knowledge base content with support for partial or full replacement."""
        logger.info(f"VECTOR STORE - Updating knowledge base {knowledgebaseId} for user {userId} (type: {update_type})")
        
        if update_type == "replace":
            # Full replacement: delete all content and add new
            try:
                # Delete all content in the knowledge base
                collection = self._get_collection(userId)
                if collection:
                    metadata_filter = {"knowledgebaseId": knowledgebaseId}
                    formatted_filter = self._format_where_filter(metadata_filter)
                    
                    results = collection.get(where=formatted_filter)
                    if results and results['ids']:
                        collection.delete(ids=results['ids'])
                        logger.info(f"VECTOR STORE - Deleted {len(results['ids'])} existing items from knowledge base")
                
                # Convert updates to messages and add them
                messages = []
                for update in updates:
                    message_data = {
                        "id": update.get('id', str(uuid.uuid4())),
                        "body": update.get('body', ''),
                        "role": update.get('role', 'knowledge_content'),
                        "messageType": update.get('messageType', 'knowledge'),
                        "contentType": update.get('contentType', 'text/plain'),
                        "source": update.get('source', 'knowledge_base'),
                        "conversationId": f"kb_{knowledgebaseId}",
                        "knowledgebaseId": knowledgebaseId,
                        "userId": userId,
                        "direction": "inbound",
                        "dateAdded": datetime.now(timezone.utc).isoformat(),
                        "locationId": "default"
                    }
                    messages.append(message_data)
                
                success = await self.add_chat_messages(
                    userId=userId,
                    messages=messages,
                    knowledgebaseId=knowledgebaseId,
                    model_name=model_name
                )
                
                return {
                    "updated": [msg['id'] for msg in messages] if success else [],
                    "failed": [] if success else [msg['id'] for msg in messages],
                    "errors": [] if success else ["Failed to add replacement content"]
                }
                
            except Exception as e:
                logger.error(f"VECTOR STORE - Failed to replace knowledge base content: {str(e)}")
                return {
                    "updated": [],
                    "failed": [update.get('id', 'unknown') for update in updates],
                    "errors": [str(e)]
                }
        else:
            # Partial update: update individual items
            return await self.update_messages_bulk(
                userId=userId,
                updates=updates,
                knowledgebaseId=knowledgebaseId,
                model_name=model_name
            )
