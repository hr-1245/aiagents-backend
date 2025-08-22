from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, timezone

from ...ai.agents.summarize_chat_agent import create_summary_graph
from ...ai.models.models import ConversationSummary, SummaryRequest
from ...ai.vector.vector_store import VectorStoreService
from ...ai.utils.langgraph_utils import invoke_graph_safely
from ..config import logger, API_REQUESTS, API_ERRORS, API_LATENCY, get_default_vector_config
from ..security import authenticate, check_rate_limit

# Create the router for summary endpoints
router = APIRouter()


@router.post("/summary", response_model=ConversationSummary)
async def get_conversation_summary(
    request: SummaryRequest,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Generate a summary of the conversation."""
    try:
        with API_LATENCY.labels('/conversation/summary').time():
            API_REQUESTS.labels('/conversation/summary').inc()
            logger.info(f"Generating summary for conversation: {request.conversationId}")

            knowledgebaseId = request.filters.get("knowledgebaseId") if request.filters else request.conversationId
            knowledgebaseId = knowledgebaseId or request.conversationId

            vector_service = VectorStoreService(config=get_default_vector_config())

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
                
                logger.info(f"Training status check - ConversationId: {request.conversationId}, MessageCount: {message_count}")
                
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
                logger.warning(f"Conversation not trained - ConversationId: {request.conversationId}, UserId: {request.userId}")
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
                logger.warning(f"No results returned despite message count > 0 - ConversationId: {request.conversationId}, MessageCount: {message_count}")
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

            logger.info(f"Summary generated successfully - ConversationId: {request.conversationId}, MessageCount: {len(results)}, SummaryLength: {len(summary)}")

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