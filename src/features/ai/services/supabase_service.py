import os
import logging
from typing import Optional, Dict, Any, Tuple, List
import httpx
import asyncio
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class SupabaseStorageService:
    """Service for interacting with Supabase storage."""
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
        self.supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")
        self.default_bucket = os.getenv("SUPABASE_DEFAULT_BUCKET", "documents")  # Default bucket for files
        
        if not all([self.supabase_url, self.supabase_anon_key]):
            logger.warning("Supabase credentials not configured. File training from Supabase will not work.")
    
    def _get_headers(self, use_service_key: bool = False) -> Dict[str, str]:
        """Get headers for Supabase API requests."""
        key = self.supabase_service_key if use_service_key and self.supabase_service_key else self.supabase_anon_key
        return {
            "Authorization": f"Bearer {key}",
            "apikey": key,
            "Content-Type": "application/json"
        }
    
    async def download_file(self, bucket: str, file_path: str) -> Tuple[bytes, str]:
        """
        Download a file from Supabase storage.
        
        Returns:
            Tuple of (file_content, content_type)
        """
        if not self.supabase_url:
            raise ValueError("Supabase URL not configured")
        
        # Construct the storage URL
        storage_url = f"{self.supabase_url}/storage/v1/object/{bucket}/{file_path}"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    storage_url,
                    headers=self._get_headers(use_service_key=True),
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    content_type = response.headers.get("content-type", "application/octet-stream")
                    return response.content, content_type
                elif response.status_code == 404:
                    raise FileNotFoundError(f"File not found: {bucket}/{file_path}")
                else:
                    raise Exception(f"Failed to download file: HTTP {response.status_code} - {response.text}")
                    
        except httpx.TimeoutException:
            raise Exception("Timeout while downloading file from Supabase")
        except Exception as e:
            logger.error(f"Error downloading file from Supabase: {e}")
            raise
    
    async def get_file_metadata(self, bucket: str, file_path: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a file in Supabase storage."""
        if not self.supabase_url:
            raise ValueError("Supabase URL not configured")
        
        # Use the list API to get file metadata
        list_url = f"{self.supabase_url}/storage/v1/object/list/{bucket}"
        
        try:
            async with httpx.AsyncClient() as client:
                # Get the directory path and filename
                path_parts = file_path.split('/')
                filename = path_parts[-1]
                folder_path = '/'.join(path_parts[:-1]) if len(path_parts) > 1 else ""
                
                response = await client.post(
                    list_url,
                    headers=self._get_headers(use_service_key=True),
                    json={
                        "limit": 100,
                        "offset": 0,
                        "sortBy": {"column": "name", "order": "asc"},
                        "prefix": folder_path
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    files = response.json()
                    # Find the specific file
                    for file_info in files:
                        if file_info.get("name") == filename:
                            return file_info
                    return None
                else:
                    logger.warning(f"Failed to get file metadata: HTTP {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting file metadata: {e}")
            return None
    
    def extract_text_from_content(self, content: bytes, content_type: str, filename: str) -> str:
        """Extract text content from file bytes based on content type."""
        try:
            # Handle text files
            if content_type.startswith("text/") or filename.endswith((".txt", ".md", ".csv")):
                return content.decode('utf-8')
            
            # Handle PDF files
            elif content_type == "application/pdf" or filename.endswith(".pdf"):
                try:
                    import PyPDF2
                    import io
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                    text_content = ""
                    for page in pdf_reader.pages:
                        text_content += page.extract_text() + "\n"
                    return text_content
                except ImportError:
                    raise Exception("PyPDF2 not installed. Cannot process PDF files.")
                except Exception as e:
                    raise Exception(f"Failed to extract text from PDF: {str(e)}")
            
            # Handle Word documents
            elif content_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"] or filename.endswith((".docx", ".doc")):
                try:
                    import docx
                    import io
                    doc = docx.Document(io.BytesIO(content))
                    text_content = ""
                    for paragraph in doc.paragraphs:
                        text_content += paragraph.text + "\n"
                    return text_content
                except ImportError:
                    raise Exception("python-docx not installed. Cannot process Word documents.")
                except Exception as e:
                    raise Exception(f"Failed to extract text from Word document: {str(e)}")
            
            # Try to decode as text as fallback
            else:
                try:
                    return content.decode('utf-8')
                except UnicodeDecodeError:
                    raise Exception(f"Unsupported file type: {content_type}")
                    
        except Exception as e:
            logger.error(f"Error extracting text from file: {e}")
            raise
    
    def chunk_text(self, text: str, chunk_size: int = 2000, overlap: int = 200) -> list:
        """Split text into overlapping chunks for better vector storage."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                sentence_end = text.rfind('.', start + chunk_size - 200, end)
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap if end < len(text) else end
        
        return chunks
    
    async def get_file_info_by_id(self, file_id: str, bucket: Optional[str] = None) -> Dict[str, Any]:
        """
        Get complete file information by file ID.
        
        Args:
            file_id: The file ID/path in Supabase storage
            bucket: Optional bucket name, uses default if not provided
            
        Returns:
            Dict containing file metadata including name, size, content_type, etc.
        """
        bucket = bucket or self.default_bucket
        
        try:
            # First try to get file metadata
            metadata = await self.get_file_metadata(bucket, file_id)
            
            if metadata:
                return {
                    "fileId": file_id,
                    "fileName": metadata.get("name", file_id.split('/')[-1]),
                    "fileSize": metadata.get("metadata", {}).get("size"),
                    "contentType": metadata.get("metadata", {}).get("mimetype", "application/octet-stream"),
                    "bucket": bucket,
                    "lastModified": metadata.get("updated_at"),
                    "metadata": metadata
                }
            else:
                # If metadata not available, infer from file_id
                filename = file_id.split('/')[-1]
                file_ext = filename.split('.')[-1].lower() if '.' in filename else ""
                
                # Guess content type from extension
                content_type_map = {
                    'pdf': 'application/pdf',
                    'txt': 'text/plain',
                    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'doc': 'application/msword',
                    'csv': 'text/csv',
                    'json': 'application/json'
                }
                
                return {
                    "fileId": file_id,
                    "fileName": filename,
                    "fileSize": None,
                    "contentType": content_type_map.get(file_ext, "application/octet-stream"),
                    "bucket": bucket,
                    "lastModified": None,
                    "metadata": {}
                }
                
        except Exception as e:
            logger.error(f"Error getting file info for {file_id}: {e}")
            raise Exception(f"Failed to get file information: {str(e)}")
    
    async def download_file_by_id(self, file_id: str, bucket: Optional[str] = None) -> Tuple[bytes, str, Dict[str, Any]]:
        """
        Download a file by ID and return content with metadata.
        
        Returns:
            Tuple of (file_content, content_type, file_info)
        """
        bucket = bucket or self.default_bucket
        
        # Get file info first
        file_info = await self.get_file_info_by_id(file_id, bucket)
        
        # Download the file
        file_content, content_type = await self.download_file(bucket, file_id)
        
        # Use content type from download if available, otherwise from metadata
        final_content_type = content_type if content_type != "application/octet-stream" else file_info["contentType"]
        
        return file_content, final_content_type, file_info

class SupabaseService:
    """Main service for interacting with Supabase database."""
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
        self.supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not all([self.supabase_url, self.supabase_anon_key]):
            logger.warning("Supabase credentials not configured. Using mock service.")
            self.client = None
        else:
            logger.info("Supabase service initialized successfully")
            self.client = self._create_client()
    
    def _create_client(self):
        """Create Supabase client."""
        try:
            from supabase import create_client
            logger.info("Creating real Supabase client connection")
            return create_client(self.supabase_url, self.supabase_anon_key)
        except ImportError:
            logger.warning("Supabase package not installed. Using mock client for development")
            return MockSupabaseClient(self.supabase_url, self.supabase_anon_key)
    
    def _get_headers(self, use_service_key: bool = False) -> Dict[str, str]:
        """Get headers for Supabase API requests."""
        key = self.supabase_service_key if use_service_key and self.supabase_service_key else self.supabase_anon_key
        return {
            "Authorization": f"Bearer {key}",
            "apikey": key,
            "Content-Type": "application/json"
        }
    
    async def get_ai_agent(self, agent_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch an AI agent from the ai_agents table.
        
        Args:
            agent_id: UUID of the agent
            user_id: UUID of the user (for access control)
            
        Returns:
            Dict containing agent data or None if not found
        """
        if not self.supabase_url:
            logger.warning("Supabase not configured, cannot fetch AI agent")
            return None
        
        try:
            # Use direct HTTP request for now, can be replaced with official Supabase client
            async with httpx.AsyncClient() as client:
                url = f"{self.supabase_url}/rest/v1/ai_agents"
                params = {
                    "id": f"eq.{agent_id}",
                    "user_id": f"eq.{user_id}",
                    "is_active": "eq.true",
                    "select": "*"
                }
                
                response = await client.get(
                    url,
                    headers=self._get_headers(use_service_key=True),
                    params=params,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    agents = response.json()
                    if agents and len(agents) > 0:
                        agent_data = agents[0]
                        logger.info(f"Successfully fetched AI agent {agent_id} for user {user_id}")
                        return self._transform_agent_data(agent_data)
                    else:
                        logger.warning(f"AI agent {agent_id} not found for user {user_id}")
                        return None
                else:
                    logger.error(f"Failed to fetch AI agent: HTTP {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching AI agent {agent_id}: {e}")
            return None
    
    async def list_ai_agents(self, user_id: str, agent_type: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List all AI agents for a user.
        
        Args:
            user_id: UUID of the user
            agent_type: Optional agent type filter (integer)
            
        Returns:
            List of agent dictionaries
        """
        if not self.supabase_url:
            logger.warning("Supabase not configured, cannot list AI agents")
            return []
        
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.supabase_url}/rest/v1/ai_agents"
                params = {
                    "user_id": f"eq.{user_id}",
                    "is_active": "eq.true",
                    "select": "*",
                    "order": "created_at.desc"
                }
                
                if agent_type is not None:
                    params["type"] = f"eq.{agent_type}"
                
                response = await client.get(
                    url,
                    headers=self._get_headers(use_service_key=True),
                    params=params,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    agents = response.json()
                    logger.info(f"Successfully fetched {len(agents)} AI agents for user {user_id}")
                    return [self._transform_agent_data(agent) for agent in agents]
                else:
                    logger.error(f"Failed to list AI agents: HTTP {response.status_code} - {response.text}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error listing AI agents for user {user_id}: {e}")
            return []
    
    async def create_ai_agent(self, agent_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create a new AI agent in the ai_agents table.
        
        Args:
            agent_data: Dictionary containing agent data
            
        Returns:
            Created agent data or None if failed
        """
        if not self.supabase_url:
            logger.warning("Supabase not configured, cannot create AI agent")
            return None
        
        try:
            # Transform data to match database schema
            db_data = self._transform_agent_for_db(agent_data)
            
            async with httpx.AsyncClient() as client:
                url = f"{self.supabase_url}/rest/v1/ai_agents"
                
                response = await client.post(
                    url,
                    headers=self._get_headers(use_service_key=True),
                    json=db_data,
                    timeout=10.0
                )
                
                if response.status_code == 201:
                    created_agent = response.json()
                    if isinstance(created_agent, list) and len(created_agent) > 0:
                        agent = created_agent[0]
                    else:
                        agent = created_agent
                    
                    logger.info(f"Successfully created AI agent {agent.get('id')} for user {agent_data.get('user_id')}")
                    return self._transform_agent_data(agent)
                else:
                    logger.error(f"Failed to create AI agent: HTTP {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error creating AI agent: {e}")
            return None
    
    async def update_ai_agent(self, agent_id: str, user_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update an AI agent in the ai_agents table.
        
        Args:
            agent_id: UUID of the agent
            user_id: UUID of the user (for access control)
            updates: Dictionary containing updated data
            
        Returns:
            Updated agent data or None if failed
        """
        if not self.supabase_url:
            logger.warning("Supabase not configured, cannot update AI agent")
            return None
        
        try:
            # Transform updates to match database schema
            db_updates = self._transform_agent_for_db(updates)
            db_updates["updated_at"] = datetime.now(timezone.utc).isoformat()
            
            async with httpx.AsyncClient() as client:
                url = f"{self.supabase_url}/rest/v1/ai_agents"
                params = {
                    "id": f"eq.{agent_id}",
                    "user_id": f"eq.{user_id}"
                }
                
                response = await client.patch(
                    url,
                    headers=self._get_headers(use_service_key=True),
                    params=params,
                    json=db_updates,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    updated_agents = response.json()
                    if updated_agents and len(updated_agents) > 0:
                        logger.info(f"Successfully updated AI agent {agent_id}")
                        return self._transform_agent_data(updated_agents[0])
                    else:
                        logger.warning(f"No agent found to update: {agent_id}")
                        return None
                else:
                    logger.error(f"Failed to update AI agent: HTTP {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error updating AI agent {agent_id}: {e}")
            return None
    
    async def delete_ai_agent(self, agent_id: str, user_id: str) -> bool:
        """
        Delete (deactivate) an AI agent in the ai_agents table.
        
        Args:
            agent_id: UUID of the agent
            user_id: UUID of the user (for access control)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.supabase_url:
            logger.warning("Supabase not configured, cannot delete AI agent")
            return False
        
        try:
            # Soft delete by setting is_active to false
            updates = {
                "is_active": False,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            async with httpx.AsyncClient() as client:
                url = f"{self.supabase_url}/rest/v1/ai_agents"
                params = {
                    "id": f"eq.{agent_id}",
                    "user_id": f"eq.{user_id}"
                }
                
                response = await client.patch(
                    url,
                    headers=self._get_headers(use_service_key=True),
                    params=params,
                    json=updates,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    logger.info(f"Successfully deleted AI agent {agent_id}")
                    return True
                else:
                    logger.error(f"Failed to delete AI agent: HTTP {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error deleting AI agent {agent_id}: {e}")
            return False
    
    def _transform_agent_data(self, db_agent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform database agent data to application format.
        
        Args:
            db_agent: Agent data from database
            
        Returns:
            Transformed agent data for application use
        """
        try:
            # Extract configuration and data fields
            configuration = db_agent.get("configuration", {})
            data = db_agent.get("data", {})
            metadata = db_agent.get("metadata", {})
            
            # Map database fields to application fields
            transformed = {
                "id": db_agent.get("id"),
                "userId": db_agent.get("user_id"),
                "name": db_agent.get("name"),
                "description": db_agent.get("description"),
                "agentType": self._map_agent_type_from_db(db_agent.get("type")),
                "isActive": db_agent.get("is_active", True),
                "createdAt": db_agent.get("created_at"),
                "updatedAt": db_agent.get("updated_at"),
                
                # Extract structured components from data field
                "personality": data.get("personality", ""),
                "intent": data.get("intent", ""),
                "additionalInformation": data.get("additionalInformation", ""),
                "variables": data.get("variables", {}),
                
                # System prompt and configuration
                "customPrompt": db_agent.get("system_prompt"),
                "configuration": configuration,
                "metadata": metadata,
                "knowledgeBaseIds": db_agent.get("knowledge_base_ids", [])
            }
            
            return transformed
            
        except Exception as e:
            logger.error(f"Error transforming agent data: {e}")
            return db_agent
    
    def _transform_agent_for_db(self, app_agent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform application agent data to database format.
        
        Args:
            app_agent: Agent data from application
            
        Returns:
            Transformed agent data for database storage
        """
        try:
            # Prepare data field with structured components
            data = {
                "personality": app_agent.get("personality", ""),
                "intent": app_agent.get("intent", ""),
                "additionalInformation": app_agent.get("additionalInformation", ""),
                "variables": app_agent.get("variables", {})
            }
            
            # Map application fields to database fields
            transformed = {
                "user_id": app_agent.get("userId"),
                "name": app_agent.get("name"),
                "description": app_agent.get("description"),
                "type": self._map_agent_type_to_db(app_agent.get("agentType")),
                "system_prompt": app_agent.get("customPrompt"),
                "configuration": app_agent.get("configuration", {}),
                "data": data,
                "metadata": app_agent.get("metadata", {}),
                "knowledge_base_ids": app_agent.get("knowledgeBaseIds", []),
                "is_active": app_agent.get("isActive", True)
            }
            
            # Remove None values
            return {k: v for k, v in transformed.items() if v is not None}
            
        except Exception as e:
            logger.error(f"Error transforming agent for database: {e}")
            return app_agent
    
    def _map_agent_type_to_db(self, agent_type: str) -> int:
        """Map string agent type to database integer type."""
        type_mapping = {
            "query": 1,
            "suggestions": 2,
            "response": 3,
            "custom": 4
        }
        return type_mapping.get(agent_type, 4)  # Default to custom
    
    def _map_agent_type_from_db(self, db_type: int) -> str:
        """Map database integer type to string agent type."""
        type_mapping = {
            1: "query",
            2: "suggestions", 
            3: "response",
            4: "custom"
        }
        return type_mapping.get(db_type, "custom")  # Default to custom

class MockSupabaseClient:
    """Mock Supabase client for development/testing purposes."""
    
    def __init__(self, url: str, key: str):
        self.url = url
        self.key = key
    
    def table(self, table_name: str):
        return MockSupabaseTable(table_name, self.url, self.key)

class MockSupabaseTable:
    """Mock Supabase table operations."""
    
    def __init__(self, table_name: str, url: str, key: str):
        self.table_name = table_name
        self.url = url
        self.key = key
        self._query = {}
        self._data = None
    
    def select(self, columns: str = "*"):
        """Mock select operation."""
        return self
    
    def insert(self, data: dict):
        """Mock insert operation."""
        self._data = data
        return self
    
    def update(self, data: dict):
        """Mock update operation."""
        self._data = data
        return self
    
    def eq(self, column: str, value):
        """Mock equality filter."""
        self._query[column] = value
        return self
    
    def lte(self, column: str, value):
        """Mock less than or equal filter."""
        self._query[f"{column}__lte"] = value
        return self
    
    def is_(self, column: str, value):
        """Mock is filter."""
        self._query[f"{column}__is"] = value
        return self
    
    async def execute(self):
        """Mock execute operation."""
        # For now, just return a mock response
        # In a real implementation, this would make HTTP requests to Supabase
        if self._data:
            return MockSupabaseResponse([{**self._data, "id": "mock-id-123"}])
        else:
            return MockSupabaseResponse([])

class MockSupabaseResponse:
    """Mock Supabase response."""
    
    def __init__(self, data: list):
        self.data = data

# Global service instance
supabase_service = SupabaseStorageService() 