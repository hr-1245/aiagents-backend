import os
from fastapi import APIRouter, HTTPException
from uuid import uuid4
from datetime import datetime, timezone
import structlog
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram

from ..ai.agents.query_agent import create_query_graph
from ..ai.utils.conversation_utils import (
    format_conversation_context, build_suggestion_prompt,
    build_response_prompt, extract_numbered_items,
    calculate_autopilot_confidence,
    FALLBACK_SUGGESTIONS, FALLBACK_RESPONSES
)
from ..ai.agents.summarize_chat_agent import create_summary_graph
from ..ai.agents.response_suggestions_agent import create_response_suggestions_graph
from ..ai.llms.graph import create_graph
from ..ai.models.models import MessageResponse, MessageRequest, QueryResponse, QueryRequest, ConversationSummary, \
    SummaryRequest, TrainingStatus, SuggestionRequest, SuggestionResponse, ResponseSuggestionRequest, ResponseSuggestionResponse
from ..ai.vector.vector_store import VectorStoreService, VectorStoreConfig
from ..ai.utils.langgraph_utils import invoke_graph_safely
from ..ai.utils.prompts import suggestion_prompt, suggestion_parser
from langchain_openai import ChatOpenAI
from ..ai.config import get_llm_config


# Configure structured logging
logger = structlog.get_logger(__name__)

# Configure metrics for API monitoring
API_REQUESTS = Counter('api_requests_total', 'Total API requests', ['endpoint'])
API_ERRORS = Counter('api_errors_total', 'Total API errors', ['endpoint'])
API_LATENCY = Histogram('api_latency_seconds', 'API latency in seconds', ['endpoint'])

# Create the router for the conversation API endpoints
router = APIRouter()

# Default vector store configuration
default_config = VectorStoreConfig(
    voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
    model_name="voyage-3-large",
    embedding_dimension=1536,  # Updated dimension for voyage-3-large
    persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
)


@router.post("/train", response_model=MessageResponse)
async def train_conversation(request: MessageRequest):
    """Train a conversation by adding messages to the vector store."""
    try:
        with API_LATENCY.labels('/conversation/train').time():
            API_REQUESTS.labels('/conversation/train').inc()
            logger.info(
                "Training conversation",
                conversationId=request.conversationId,
                message_count=len(request.messages)
            )

            knowledgebaseId = request.knowledgebaseId or request.conversationId
            cleaned_messages = []

            for msg in request.messages:
                if not any([msg.body, msg.role, msg.messageType]):
                    continue

                try:
                    msg_data = msg.dict()
                    msg_data.update({
                        "userId": request.userId,
                        "conversationId": request.conversationId,
                        "locationId": request.locationId or "default",
                        "knowledgebaseId": knowledgebaseId
                    })
                    cleaned_messages.append(msg_data)
                except Exception as e:
                    logger.error(f"Error cleaning message data: {str(e)}")
                    continue

            if not cleaned_messages:
                raise HTTPException(status_code=400, detail="No valid messages to train")

            config = default_config.copy(update={"model_name": request.model_name}) if hasattr(request,
                                                                                               "model_name") and request.model_name else default_config
            vector_service = VectorStoreService(config=config)
            success = await vector_service.add_chat_messages(
                userId=request.userId,
                messages=cleaned_messages,
                knowledgebaseId=knowledgebaseId,
                model_name=request.model_name if hasattr(request, "model_name") else None
            )

            now = datetime.now(timezone.utc)
            try:
                start_date = min([msg.get("dateAdded", now.isoformat()) for msg in cleaned_messages],
                                 default=now.isoformat())
                end_date = max([msg.get("dateAdded", now.isoformat()) for msg in cleaned_messages],
                               default=now.isoformat())
            except Exception as e:
                logger.error(f"Error calculating date range: {str(e)}")
                start_date = now.isoformat()
                end_date = now.isoformat()

            response_data = {
                "conversationId": request.conversationId,
                "messageCount": len(cleaned_messages),
                "dateRange": {
                    "start": start_date,
                    "end": end_date
                }
            }

            return MessageResponse(
                success=success,
                message=f"Successfully trained {len(cleaned_messages)} messages" if success else "Failed to train conversation",
                data=response_data
            )

    except HTTPException as he:
        API_ERRORS.labels('/conversation/train').inc()
        logger.error(f"HTTP error in train: {he.detail}")
        raise
    except Exception as e:
        API_ERRORS.labels('/conversation/train').inc()
        logger.error(f"Error training conversation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error training conversation: {str(e)}"
        )


@router.post("/query", response_model=QueryResponse)
async def query_conversation(request: QueryRequest):
    """Query conversation history and generate a response."""
    try:
        with API_LATENCY.labels('/conversation/query').time():
            API_REQUESTS.labels('/conversation/query').inc()
            logger.info(
                "Querying conversation",
                conversationId=request.conversationId,
                query=request.query
            )

            knowledgebaseId = request.knowledgebaseId or request.conversationId

            retrieved_context_messages = []
            try:
                config = default_config.copy(update={"model_name": request.model_name}) if hasattr(request,
                                                                                                   "model_name") and request.model_name else default_config
                vector_service = VectorStoreService(config=config)
                
                # Debug: Check if conversation exists in vector store
                message_count = await vector_service.get_message_count(
                    userId=request.userId,
                    metadata_filter={
                        "conversationId": request.conversationId,
                        "knowledgebaseId": knowledgebaseId
                    },
                    knowledgebaseId=knowledgebaseId
                )
                
                logger.info(
                    "Vector store check",
                    userId=request.userId,
                    conversationId=request.conversationId,
                    knowledgebaseId=knowledgebaseId,
                    total_messages_in_store=message_count
                )
                
                retrieved_context_messages = await vector_service.query_chat_history(
                    userId=request.userId,
                    query=request.query,
                    metadata_filter={
                        "conversationId": request.conversationId,
                        "knowledgebaseId": knowledgebaseId
                    },
                    k=request.limit or 5,
                    knowledgebaseId=knowledgebaseId,
                    model_name=request.model_name if hasattr(request, "model_name") else None
                )
                
                # Add detailed logging for debugging
                logger.info(
                    "Vector search completed",
                    userId=request.userId,
                    conversationId=request.conversationId,
                    knowledgebaseId=knowledgebaseId,
                    retrieved_count=len(retrieved_context_messages),
                    query_length=len(request.query.strip()) if request.query else 0
                )
                
            except Exception as e:
                logger.error(f"Error querying vector store for context: {str(e)}")
                retrieved_context_messages = []

            # Improved context formatting
            if retrieved_context_messages:
                # Format as readable conversation history
                formatted_messages = []
                for msg in retrieved_context_messages:
                    role = msg.get('role', 'unknown')
                    body = msg.get('body', '')
                    timestamp = msg.get('dateAdded', '')
                    formatted_messages.append(f"[{role}] {body}")
                
                formatted_context = "\n".join(formatted_messages)
                logger.info(f"Context formatted successfully - message_count: {len(formatted_messages)}")
            else:
                formatted_context = "No relevant conversation history found."
                logger.warning(
                    "No context retrieved",
                    userId=request.userId,
                    conversationId=request.conversationId,
                    knowledgebaseId=knowledgebaseId,
                    query=request.query[:100] + "..." if len(request.query) > 100 else request.query
                )

            graph = create_query_graph()

            state = {
                "query": request.query,
                "userId": request.userId,
                "conversationId": request.conversationId,
                "knowledgebaseId": knowledgebaseId,
                "retrieved_context": formatted_context
            }

            final_state = {}
            try:
                final_state = await invoke_graph_safely(graph, state)
            except Exception as e:
                logger.error(f"Graph invocation error: {str(e)}")
                final_state = {"response": "I encountered an issue processing your request due to a graph error.", "suggestions": []}

            response_text = final_state.get("response", "No response generated by the AI.")
            suggestions = final_state.get("suggestions", [])
            
            logger.info(f"Query completed - suggestions_count: {len(suggestions)}")

            return QueryResponse(
                messages=retrieved_context_messages,
                total=len(retrieved_context_messages),
                query=request.query,
                answer=response_text,
                suggestions=suggestions,
                timestamp=datetime.now(timezone.utc)
            )

    except HTTPException as he:
        API_ERRORS.labels('/conversation/query').inc()
        logger.error(f"HTTP error in query: {he.detail}")
        raise
    except Exception as e:
        API_ERRORS.labels('/conversation/query').inc()
        logger.error(f"Error querying conversation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error querying conversation: {str(e)}"
        )


@router.post("/summary", response_model=ConversationSummary)
async def get_conversation_summary(request: SummaryRequest):
    """Generate a summary of the conversation."""
    try:
        with API_LATENCY.labels('/conversation/summary').time():
            API_REQUESTS.labels('/conversation/summary').inc()
            logger.info(f"Generating summary for conversation: {request.conversationId}")

            knowledgebaseId = request.filters.get("knowledgebaseId") if request.filters else request.conversationId
            knowledgebaseId = knowledgebaseId or request.conversationId

            vector_service = VectorStoreService(config=default_config)

            # First, check if the conversation exists and is trained
            try:
                message_count = await vector_service.get_message_count(
                    userId=request.userId,
                    metadata_filter={
                        "conversationId": request.conversationId,
                        "knowledgebaseId": knowledgebaseId
                    },
                    knowledgebaseId=knowledgebaseId
                )
                
                logger.info("Training status check", 
                           conversationId=request.conversationId,
                           messageCount=message_count)
                
            except Exception as e:
                logger.error(f"Error checking training status: {str(e)}")
                return ConversationSummary(
                    success=False,
                    summary="Unable to access conversation data. Please check your connection and try again.",
                    metadata={
                        "conversationId": request.conversationId,
                        "userId": request.userId,
                        "messageCount": 0,
                        "error": "database_access_error"
                    },
                    is_trained=False,
                    error_code="DATABASE_ACCESS_ERROR",
                    recommendations=[
                        "Check your network connection",
                        "Verify the conversation ID is correct",
                        "Try again in a few moments"
                    ],
                    timestamp=datetime.now(timezone.utc)
                )

            # If no messages found, conversation is not trained
            if message_count == 0:
                logger.warning("Conversation not trained", 
                              conversationId=request.conversationId,
                              userId=request.userId)
                return ConversationSummary(
                    success=False,
                    summary="This conversation has not been trained yet. Please train the conversation first to generate a summary.",
                    metadata={
                        "conversationId": request.conversationId,
                        "userId": request.userId,
                        "messageCount": 0,
                        "training_required": True
                    },
                    is_trained=False,
                    error_code="CONVERSATION_NOT_TRAINED",
                    recommendations=[
                        "Train this conversation using the /ai/conversation/train endpoint",
                        "Ensure you have valid conversation messages to upload",
                        "Check that your userId and conversationId are correct"
                    ],
                    timestamp=datetime.now(timezone.utc)
                )

            # Retrieve conversation messages
            try:
                results = await vector_service.get_conversation_by_id(
                    userId=request.userId,
                    conversationId=request.conversationId,
                    knowledgebaseId=knowledgebaseId
                )
            except Exception as e:
                logger.error(f"Error retrieving messages for summary: {str(e)}")
                return ConversationSummary(
                    success=False,
                    summary="Error retrieving conversation messages. Please try again.",
                    metadata={
                        "conversationId": request.conversationId,
                        "userId": request.userId,
                        "messageCount": message_count,
                        "error": "message_retrieval_error"
                    },
                    is_trained=True,
                    error_code="MESSAGE_RETRIEVAL_ERROR",
                    recommendations=[
                        "Try again in a few moments",
                        "Check if the conversation ID is correct",
                        "Contact support if the issue persists"
                    ],
                    timestamp=datetime.now(timezone.utc)
                )

            if not results:
                logger.warning("No results returned despite message count > 0",
                              conversationId=request.conversationId,
                              messageCount=message_count)
                return ConversationSummary(
                    success=False,
                    summary="Conversation data appears to be corrupted or inaccessible. Please re-train the conversation.",
                    metadata={
                        "conversationId": request.conversationId,
                        "userId": request.userId,
                        "messageCount": message_count,
                        "data_corruption": True
                    },
                    is_trained=False,
                    error_code="DATA_CORRUPTION",
                    recommendations=[
                        "Re-train this conversation with fresh data",
                        "Verify your conversation messages are valid",
                        "Contact support if the issue persists"
                    ],
                    timestamp=datetime.now(timezone.utc)
                )

            # Sort messages chronologically
            results.sort(key=lambda r: r.get("dateAdded", ""))

            # Compose conversation text with safe truncation
            try:
                messages_text = "\n".join(
                    f"{r.get('role', 'unknown')}: {r.get('body', '')}" for r in results
                )
                truncated_text = messages_text[:4000]
                if len(messages_text) > 4000:
                    truncated_text += "\n... (conversation truncated for processing)"
            except Exception as e:
                logger.error(f"Error creating messages text: {str(e)}")
                return ConversationSummary(
                    success=False,
                    summary="Error processing conversation messages. The conversation data may be corrupted.",
                    metadata={
                        "conversationId": request.conversationId,
                        "userId": request.userId,
                        "messageCount": len(results),
                        "processing_error": True
                    },
                    is_trained=True,
                    error_code="MESSAGE_PROCESSING_ERROR",
                    recommendations=[
                        "Re-train the conversation with valid message data",
                        "Check message format and content",
                        "Contact support if the issue persists"
                    ],
                    timestamp=datetime.now(timezone.utc)
                )

            # Generate summary using AI
            graph = create_summary_graph()
            state = {
                "retrieved_context": truncated_text,
                "userId": request.userId,
                "conversationId": request.conversationId,
                "knowledgebaseId": knowledgebaseId
            }

            try:
                final_state = await invoke_graph_safely(graph, state)
                summary = final_state.get("summary", "")
                
                if not summary or summary.strip() == "":
                    summary = "Unable to generate a meaningful summary from the conversation content."
                    
            except Exception as e:
                logger.error(f"Graph invocation error during summary generation: {str(e)}")
                return ConversationSummary(
                    success=False,
                    summary="Error generating summary using AI. The service may be temporarily unavailable.",
                    metadata={
                        "conversationId": request.conversationId,
                        "userId": request.userId,
                        "messageCount": len(results),
                        "ai_error": True
                    },
                    is_trained=True,
                    error_code="AI_SERVICE_ERROR",
                    recommendations=[
                        "Try again in a few moments",
                        "The AI service may be temporarily unavailable",
                        "Check your OpenAI API configuration"
                    ],
                    timestamp=datetime.now(timezone.utc)
                )

            # Build metadata
            now = datetime.now(timezone.utc)
            try:
                timestamps = [r.get("dateAdded") for r in results if r.get("dateAdded")]
                start_date = min(timestamps) if timestamps else now.isoformat()
                end_date = max(timestamps) if timestamps else now.isoformat()
            except Exception as e:
                logger.error(f"Error calculating date range: {str(e)}")
                start_date = end_date = now.isoformat()

            # Count message types for additional insights
            message_roles = {}
            for r in results:
                role = r.get("role", "unknown")
                message_roles[role] = message_roles.get(role, 0) + 1

            metadata = {
                "conversationId": request.conversationId,
                "userId": request.userId,
                "messageCount": len(results),
                "dateRange": {"start": start_date, "end": end_date},
                "messageTypes": message_roles,
                "summaryLength": len(summary),
                "conversationLength": len(truncated_text),
                "wasTruncated": len(messages_text) > 4000
            }

            logger.info("Summary generated successfully",
                       conversationId=request.conversationId,
                       messageCount=len(results),
                       summaryLength=len(summary))

            return ConversationSummary(
                success=True,
                summary=summary,
                metadata=metadata,
                is_trained=True,
                error_code=None,
                recommendations=[
                    "This summary was generated successfully",
                    "You can use the suggestions endpoint for follow-up questions",
                    "Train with more messages for better summaries"
                ],
                timestamp=datetime.now(timezone.utc)
            )

    except HTTPException as he:
        API_ERRORS.labels('/conversation/summary').inc()
        logger.error(f"HTTP error in summary: {he.detail}")
        raise
    except Exception as e:
        API_ERRORS.labels('/conversation/summary').inc()
        logger.error(f"Unexpected error generating summary: {str(e)}", exc_info=True)
        
        return ConversationSummary(
            success=False,
            summary="An unexpected error occurred while generating the summary. Please try again.",
            metadata={
                "conversationId": getattr(request, 'conversationId', 'unknown'),
                "userId": getattr(request, 'userId', 'unknown'),
                "messageCount": 0,
                "unexpected_error": True
            },
            is_trained=None,
            error_code="UNEXPECTED_ERROR",
            recommendations=[
                "Try again in a few moments",
                "Check your request parameters",
                "Contact support if the issue persists"
            ],
            timestamp=datetime.now(timezone.utc)
        )


@router.post("/training-status", response_model=TrainingStatus)
async def check_training_status(request: SummaryRequest):
    """Check the training status of a conversation in the vector store."""
    try:
        with API_LATENCY.labels('/conversation/training-status').time():
            API_REQUESTS.labels('/conversation/training-status').inc()
            logger.info(f"Checking training status for conversation: {request.conversationId}")
            knowledgebaseId = request.filters.get("knowledgebaseId",
                                                  request.conversationId) if request.filters else request.conversationId

            try:
                config = default_config.copy(update={"model_name": request.model_name}) if hasattr(request,
                                                                                                   "model_name") and request.model_name else default_config
                vector_service = VectorStoreService(config=config)
                message_count = await vector_service.get_message_count(
                    userId=request.userId,
                    metadata_filter={
                        "conversationId": request.conversationId,
                        "knowledgebaseId": knowledgebaseId
                    },
                    knowledgebaseId=knowledgebaseId
                )
                latest_message = await vector_service.query_chat_history(
                    userId=request.userId,
                    query="",
                    metadata_filter={
                        "conversationId": request.conversationId,
                        "knowledgebaseId": knowledgebaseId
                    },
                    k=1,
                    knowledgebaseId=knowledgebaseId,
                    model_name=request.model_name if hasattr(request, "model_name") else None
                )
            except Exception as e:
                logger.error(f"Error querying vector store: {str(e)}")
                message_count = 0
                latest_message = []

            is_trained = message_count > 0

            try:
                last_updated = latest_message[0].get("dateAdded") if latest_message else None
            except Exception as e:
                logger.error(f"Error getting last_updated: {str(e)}")
                last_updated = None

            return TrainingStatus(
                is_trained=is_trained,
                last_updated=last_updated,
                message_count=message_count,
                vector_count=message_count
            )

    except HTTPException as he:
        API_ERRORS.labels('/conversation/training-status').inc()
        logger.error(f"HTTP error in training-status: {he.detail}")
        raise
    except Exception as e:
        API_ERRORS.labels('/conversation/training-status').inc()
        logger.error(f"Error checking training status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error checking training status: {str(e)}"
        )


@router.delete("/user/{userId}")
async def delete_user_conversations(userId: str):
    """Delete all conversation data for a user from the vector store."""
    try:
        with API_LATENCY.labels('/conversation/delete').time():
            API_REQUESTS.labels('/conversation/delete').inc()

            if not userId:
                raise HTTPException(status_code=400, detail="User ID is required")

            try:
                vector_service = VectorStoreService(config=default_config)
                success = await vector_service.delete_user_data(userId)
            except Exception as e:
                logger.error(f"Error in delete_user_data: {str(e)}")
                success = False

            if not success:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to delete user data"
                )

            return {"success": True, "message": f"All conversation data for user {userId} has been deleted."}

    except HTTPException as he:
        API_ERRORS.labels('/conversation/delete').inc()
        logger.error(f"HTTP error in delete: {he.detail}")
        raise
    except Exception as e:
        API_ERRORS.labels('/conversation/delete').inc()
        logger.error(f"Error deleting user data: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting user data: {str(e)}"
        )


@router.delete("/conversation/{conversationId}")
async def delete_specific_conversation(conversationId: str, userId: str, knowledgebaseId: Optional[str] = None):
    """Delete a specific conversation from the vector store."""
    try:
        with API_LATENCY.labels('/conversation/delete-specific').time():
            API_REQUESTS.labels('/conversation/delete-specific').inc()

            if not userId:
                raise HTTPException(status_code=400, detail="User ID is required")

            if not conversationId:
                raise HTTPException(status_code=400, detail="Conversation ID is required")

            try:
                vector_service = VectorStoreService(config=default_config)
                success = await vector_service.delete_conversation(
                    userId=userId,
                    conversationId=conversationId,
                    knowledgebaseId=knowledgebaseId
                )
            except Exception as e:
                logger.error(f"Error in delete_conversation: {str(e)}")
                success = False

            if not success:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to delete conversation"
                )

            return {
                "success": True,
                "message": f"Conversation {conversationId} has been deleted."
            }

    except HTTPException as he:
        API_ERRORS.labels('/conversation/delete-specific').inc()
        logger.error(f"HTTP error in delete-specific: {he.detail}")
        raise
    except Exception as e:
        API_ERRORS.labels('/conversation/delete-specific').inc()
        logger.error(f"Error deleting conversation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting conversation: {str(e)}"
        )


@router.get("/debug/conversation/{conversationId}")
async def debug_conversation(conversationId: str, userId: str, knowledgebaseId: Optional[str] = None):
    """Debug endpoint to check if a conversation exists in the vector store."""
    try:
        API_REQUESTS.labels('/conversation/debug').inc()
        logger.info(f"Debug check for conversation - ConversationId: {conversationId}, UserId: {userId}")

        knowledgebaseId = knowledgebaseId or conversationId
        vector_service = VectorStoreService(config=default_config)
        
        # Get all messages for the conversation
        all_messages = await vector_service.get_conversation_by_id(
            userId=userId,
            conversationId=conversationId,
            knowledgebaseId=knowledgebaseId
        )
        
        # Get message count
        message_count = await vector_service.get_message_count(
            userId=userId,
            metadata_filter={
                "conversationId": conversationId,
                "knowledgebaseId": knowledgebaseId
            },
            knowledgebaseId=knowledgebaseId
        )
        
        return {
            "conversationId": conversationId,
            "userId": userId,
            "knowledgebaseId": knowledgebaseId,
            "messageCount": message_count,
            "messagesFound": len(all_messages),
            "messages": all_messages[:5] if all_messages else [],  # Show first 5 messages
            "exists": message_count > 0
        }
        
    except Exception as e:
        API_ERRORS.labels('/conversation/debug').inc()
        logger.error(f"Error in debug endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error checking conversation: {str(e)}"
        )


@router.post("/suggestions", response_model=SuggestionResponse)
async def generate_suggestions(request: SuggestionRequest):
    """Generate follow-up questions for a conversation."""
    try:
        vector_service = VectorStoreService(config=default_config)
        messages = request.recentMessages or []

        if not messages:
            try:
                results = await vector_service.get_conversation_by_id(
                    userId=request.userId,
                    conversationId=request.conversationId,
                    knowledgebaseId=request.knowledgebaseId or request.conversationId
                )
                results.sort(key=lambda r: r.get("dateAdded", ""))
                messages = results[-5:]
            except Exception:
                messages = []

        if not messages:
            return SuggestionResponse(
                suggestions=FALLBACK_SUGGESTIONS,
                total=3,
                conversationId=request.conversationId
            )

        conversation_text = format_conversation_context(messages)
        llm = ChatOpenAI(**get_llm_config())

        prompt = build_suggestion_prompt(
            context=conversation_text,
            additional_context=request.context,
            limit=3
        )
        result = await llm.ainvoke(prompt)
        suggestions = extract_numbered_items(result.content or "", limit=3)

        if not suggestions:
            suggestions = FALLBACK_SUGGESTIONS

        return SuggestionResponse(
            suggestions=suggestions,
            total=len(suggestions),
            conversationId=request.conversationId
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Suggestion generation failed: {str(e)}")


@router.post("/response-suggestions", response_model=ResponseSuggestionResponse)
async def generate_response_suggestions(request: ResponseSuggestionRequest):
    """Generate 3 response options for the next best response based on conversation context."""
    try:
        vector_service = VectorStoreService(config=default_config)
        messages = request.recentMessages or []

        if not messages:
            try:
                results = await vector_service.get_conversation_by_id(
                    userId=request.userId,
                    conversationId=request.conversationId,
                    knowledgebaseId=request.knowledgebaseId or request.conversationId
                )
                results.sort(key=lambda r: r.get("dateAdded", ""))
                messages = results[-5:]
            except Exception:
                messages = []

        last_customer_msg = request.lastCustomerMessage
        if not last_customer_msg:
            for msg in reversed(messages):
                if msg.get("role") == "customer" or msg.get("direction") == "inbound":
                    last_customer_msg = msg.get("body", "")
                    break

        if not last_customer_msg:
            fallback_response = FALLBACK_RESPONSES[0]
            
            response_data = {
                "response_suggestion": fallback_response,
                "response_suggestions": [fallback_response],
                "total": 1,
                "conversationId": request.conversationId
            }
            
            if request.autopilot:
                response_data["autopilot_response"] = fallback_response
                response_data["confidence_score"] = 0.5  # Low confidence for fallback
            
            return ResponseSuggestionResponse(**response_data)

        context = format_conversation_context(messages)
        llm = ChatOpenAI(**get_llm_config())

        prompt = build_response_prompt(
            context=context,
            last_message=last_customer_msg,
            additional_context=request.context,
            autopilot=request.autopilot
        )

        result = await llm.ainvoke(prompt)
        
        # Always return single best response
        best_response = (result.content or "").strip()
        if not best_response:
            best_response = FALLBACK_RESPONSES[0]
        
        response_data = {
            "response_suggestion": best_response,
            "response_suggestions": [best_response],
            "total": 1,
            "conversationId": request.conversationId
        }
        
        # Add autopilot fields if requested
        if request.autopilot:
            confidence = calculate_autopilot_confidence(
                context,
                last_customer_msg,
                best_response
            )
            response_data["autopilot_response"] = best_response
            response_data["confidence_score"] = confidence

        return ResponseSuggestionResponse(**response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Response suggestion failed: {str(e)}")