from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from pydantic import BaseModel
from datetime import datetime, timezone

from ...ai.vector.vector_store import VectorStoreService
from ...ai.models.models import (
    ConversationUpdateRequest, KnowledgeBaseUpdateRequest, BulkUpdateRequest,
    UpdateResponse, BulkUpdateResponse, MessageUpdate
)
from ..config import logger, API_REQUESTS, API_ERRORS, API_LATENCY, get_default_vector_config
from ..security import authenticate, check_rate_limit

# Create the router for management endpoints
router = APIRouter()

# Pydantic models for conversation settings
class ConversationSettings(BaseModel):
    conversationId: str
    userId: str
    aiAgentId: Optional[str] = None
    knowledgebaseId: Optional[str] = None
    additionalKnowledgebaseIds: Optional[list] = []
    temperature: Optional[float] = 0.3
    model: Optional[str] = "gpt-4o-mini"
    humanlikeBehavior: Optional[bool] = True

class ConversationSettingsResponse(BaseModel):
    success: bool
    message: str
    settings: Optional[ConversationSettings] = None

# In-memory storage for conversation settings (in production, use database)
conversation_settings_store = {}

@router.post("/conversation/{conversationId}/settings", response_model=ConversationSettingsResponse)
async def save_conversation_settings(
    conversationId: str,
    settings: ConversationSettings,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Save AI settings for a specific conversation."""
    try:
        logger.info(f"Saving settings for conversation {conversationId}")
        
        # Validate that conversationId matches
        if settings.conversationId != conversationId:
            raise HTTPException(status_code=400, detail="Conversation ID mismatch")
        
        # Store settings (in production, save to database)
        settings_key = f"{settings.userId}:{conversationId}"
        conversation_settings_store[settings_key] = settings.dict()
        
        logger.info(f"Settings saved for conversation {conversationId}: aiAgentId={settings.aiAgentId}")
        
        return ConversationSettingsResponse(
            success=True,
            message="Conversation settings saved successfully",
            settings=settings
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save conversation settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save conversation settings: {str(e)}")

@router.get("/conversation/{conversationId}/settings", response_model=ConversationSettingsResponse)
async def get_conversation_settings(
    conversationId: str,
    userId: str,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Get AI settings for a specific conversation."""
    try:
        settings_key = f"{userId}:{conversationId}"
        
        if settings_key in conversation_settings_store:
            settings_data = conversation_settings_store[settings_key]
            settings = ConversationSettings(**settings_data)
            
            return ConversationSettingsResponse(
                success=True,
                message="Conversation settings retrieved successfully",
                settings=settings
            )
        else:
            # Return default settings if none found
            default_settings = ConversationSettings(
                conversationId=conversationId,
                userId=userId,
                aiAgentId=None,
                knowledgebaseId=conversationId,  # Default to conversation ID
                temperature=0.3,
                model="gpt-4o-mini",
                humanlikeBehavior=True
            )
            
            return ConversationSettingsResponse(
                success=True,
                message="No settings found, returning defaults",
                settings=default_settings
            )
            
    except Exception as e:
        logger.error(f"Failed to get conversation settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation settings: {str(e)}")

# ============================================================================
# UPDATE ENDPOINTS
# ============================================================================

@router.put("/conversation/{conversationId}/update", response_model=UpdateResponse)
async def update_conversation(
    conversationId: str,
    request: ConversationUpdateRequest,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Update messages in a specific conversation."""
    try:
        with API_LATENCY.labels('/conversation/update').time():
            API_REQUESTS.labels('/conversation/update').inc()
            
            logger.info(f"Updating conversation {conversationId} with {len(request.updates)} updates")
            
            # Validate conversation ID matches
            if request.conversationId != conversationId:
                raise HTTPException(status_code=400, detail="Conversation ID mismatch")
            
            if not request.updates:
                raise HTTPException(status_code=400, detail="No updates provided")
            
            vector_service = VectorStoreService(config=get_default_vector_config())
            knowledgebaseId = request.knowledgebaseId or conversationId
            
            if request.updateType == "replace":
                # Replace entire conversation content
                new_messages = []
                for update in request.updates:
                    message_data = {
                        "id": update.id,
                        "body": update.body or "",
                        "role": update.role or "user",
                        "messageType": update.messageType or "text",
                        "contentType": update.contentType or "text/plain",
                        "source": update.source or "conversation",
                        "conversationId": conversationId,
                        "knowledgebaseId": knowledgebaseId,
                        "userId": request.userId,
                        "direction": "inbound",
                        "dateAdded": datetime.now(timezone.utc).isoformat(),
                        "locationId": "default"
                    }
                    if update.metadata:
                        message_data.update(update.metadata)
                    new_messages.append(message_data)
                
                success = await vector_service.replace_conversation_content(
                    userId=request.userId,
                    conversationId=conversationId,
                    new_messages=new_messages,
                    knowledgebaseId=knowledgebaseId
                )
                
                if success:
                    return UpdateResponse(
                        success=True,
                        message=f"Successfully replaced conversation {conversationId} content",
                        updatedCount=len(request.updates),
                        failedCount=0,
                        updatedIds=[update.id for update in request.updates],
                        timestamp=datetime.now(timezone.utc)
                    )
                else:
                    raise HTTPException(status_code=500, detail="Failed to replace conversation content")
            
            else:
                # Partial updates
                update_data = []
                for update in request.updates:
                    update_dict = {"id": update.id}
                    if update.body is not None:
                        update_dict["body"] = update.body
                    if update.role is not None:
                        update_dict["role"] = update.role
                    if update.messageType is not None:
                        update_dict["messageType"] = update.messageType
                    if update.contentType is not None:
                        update_dict["contentType"] = update.contentType
                    if update.source is not None:
                        update_dict["source"] = update.source
                    if update.status is not None:
                        update_dict["status"] = update.status
                    if update.metadata:
                        update_dict["metadata"] = update.metadata
                    update_data.append(update_dict)
                
                results = await vector_service.update_messages_bulk(
                    userId=request.userId,
                    updates=update_data,
                    knowledgebaseId=knowledgebaseId
                )
                
                return UpdateResponse(
                    success=len(results["failed"]) == 0,
                    message=f"Updated {len(results['updated'])} messages, {len(results['failed'])} failed",
                    updatedCount=len(results["updated"]),
                    failedCount=len(results["failed"]),
                    errors=results["errors"],
                    updatedIds=results["updated"],
                    timestamp=datetime.now(timezone.utc)
                )
    
    except HTTPException:
        API_ERRORS.labels('/conversation/update').inc()
        raise
    except Exception as e:
        API_ERRORS.labels('/conversation/update').inc()
        logger.error(f"Error updating conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error updating conversation: {str(e)}")

@router.put("/knowledge-base/{knowledgebaseId}/update", response_model=UpdateResponse)
async def update_knowledge_base(
    knowledgebaseId: str,
    request: KnowledgeBaseUpdateRequest,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Update content in a specific knowledge base."""
    try:
        with API_LATENCY.labels('/knowledge-base/update').time():
            API_REQUESTS.labels('/knowledge-base/update').inc()
            
            logger.info(f"Updating knowledge base {knowledgebaseId} with {len(request.updates)} updates")
            
            # Validate knowledge base ID matches
            if request.knowledgebaseId != knowledgebaseId:
                raise HTTPException(status_code=400, detail="Knowledge base ID mismatch")
            
            if not request.updates:
                raise HTTPException(status_code=400, detail="No updates provided")
            
            vector_service = VectorStoreService(config=get_default_vector_config())
            
            # Prepare update data
            update_data = []
            for update in request.updates:
                update_dict = {"id": update.id}
                if update.body is not None:
                    update_dict["body"] = update.body
                if update.role is not None:
                    update_dict["role"] = update.role
                if update.messageType is not None:
                    update_dict["messageType"] = update.messageType
                if update.contentType is not None:
                    update_dict["contentType"] = update.contentType
                if update.source is not None:
                    update_dict["source"] = update.source
                if update.status is not None:
                    update_dict["status"] = update.status
                if update.metadata:
                    update_dict["metadata"] = update.metadata
                update_data.append(update_dict)
            
            results = await vector_service.update_knowledge_base_content(
                userId=request.userId,
                knowledgebaseId=knowledgebaseId,
                updates=update_data,
                update_type=request.updateType
            )
            
            return UpdateResponse(
                success=len(results["failed"]) == 0,
                message=f"Updated {len(results['updated'])} items, {len(results['failed'])} failed",
                updatedCount=len(results["updated"]),
                failedCount=len(results["failed"]),
                errors=results["errors"],
                updatedIds=results["updated"],
                timestamp=datetime.now(timezone.utc)
            )
    
    except HTTPException:
        API_ERRORS.labels('/knowledge-base/update').inc()
        raise
    except Exception as e:
        API_ERRORS.labels('/knowledge-base/update').inc()
        logger.error(f"Error updating knowledge base: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error updating knowledge base: {str(e)}")

@router.put("/bulk-update", response_model=BulkUpdateResponse)
async def bulk_update(
    request: BulkUpdateRequest,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Perform bulk updates across multiple conversations and knowledge bases."""
    try:
        with API_LATENCY.labels('/bulk-update').time():
            API_REQUESTS.labels('/bulk-update').inc()
            
            logger.info(f"Bulk update: {len(request.conversationUpdates)} conversations, {len(request.knowledgeBaseUpdates)} knowledge bases")
            
            conversation_results = []
            knowledge_base_results = []
            total_updated = 0
            total_failed = 0
            
            vector_service = VectorStoreService(config=get_default_vector_config())
            
            # Process conversation updates
            for conv_update in request.conversationUpdates:
                try:
                    if conv_update.updateType == "replace":
                        # Replace conversation content
                        new_messages = []
                        for update in conv_update.updates:
                            message_data = {
                                "id": update.id,
                                "body": update.body or "",
                                "role": update.role or "user",
                                "messageType": update.messageType or "text",
                                "contentType": update.contentType or "text/plain",
                                "source": update.source or "conversation",
                                "conversationId": conv_update.conversationId,
                                "knowledgebaseId": conv_update.knowledgebaseId or conv_update.conversationId,
                                "userId": conv_update.userId,
                                "direction": "inbound",
                                "dateAdded": datetime.now(timezone.utc).isoformat(),
                                "locationId": "default"
                            }
                            if update.metadata:
                                message_data.update(update.metadata)
                            new_messages.append(message_data)
                        
                        success = await vector_service.replace_conversation_content(
                            userId=conv_update.userId,
                            conversationId=conv_update.conversationId,
                            new_messages=new_messages,
                            knowledgebaseId=conv_update.knowledgebaseId
                        )
                        
                        if success:
                            result = UpdateResponse(
                                success=True,
                                message=f"Successfully replaced conversation {conv_update.conversationId}",
                                updatedCount=len(conv_update.updates),
                                failedCount=0,
                                updatedIds=[update.id for update in conv_update.updates],
                                timestamp=datetime.now(timezone.utc)
                            )
                            total_updated += len(conv_update.updates)
                        else:
                            result = UpdateResponse(
                                success=False,
                                message=f"Failed to replace conversation {conv_update.conversationId}",
                                updatedCount=0,
                                failedCount=len(conv_update.updates),
                                errors=["Conversation replacement failed"],
                                timestamp=datetime.now(timezone.utc)
                            )
                            total_failed += len(conv_update.updates)
                    else:
                        # Partial updates
                        update_data = []
                        for update in conv_update.updates:
                            update_dict = {"id": update.id}
                            for field in ['body', 'role', 'messageType', 'contentType', 'source', 'status']:
                                if getattr(update, field) is not None:
                                    update_dict[field] = getattr(update, field)
                            if update.metadata:
                                update_dict["metadata"] = update.metadata
                            update_data.append(update_dict)
                        
                        results = await vector_service.update_messages_bulk(
                            userId=conv_update.userId,
                            updates=update_data,
                            knowledgebaseId=conv_update.knowledgebaseId or conv_update.conversationId
                        )
                        
                        result = UpdateResponse(
                            success=len(results["failed"]) == 0,
                            message=f"Conversation {conv_update.conversationId}: {len(results['updated'])} updated, {len(results['failed'])} failed",
                            updatedCount=len(results["updated"]),
                            failedCount=len(results["failed"]),
                            errors=results["errors"],
                            updatedIds=results["updated"],
                            timestamp=datetime.now(timezone.utc)
                        )
                        total_updated += len(results["updated"])
                        total_failed += len(results["failed"])
                    
                    conversation_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error updating conversation {conv_update.conversationId}: {e}")
                    error_result = UpdateResponse(
                        success=False,
                        message=f"Error updating conversation {conv_update.conversationId}",
                        updatedCount=0,
                        failedCount=len(conv_update.updates),
                        errors=[str(e)],
                        timestamp=datetime.now(timezone.utc)
                    )
                    conversation_results.append(error_result)
                    total_failed += len(conv_update.updates)
            
            # Process knowledge base updates
            for kb_update in request.knowledgeBaseUpdates:
                try:
                    update_data = []
                    for update in kb_update.updates:
                        update_dict = {"id": update.id}
                        for field in ['body', 'role', 'messageType', 'contentType', 'source', 'status']:
                            if getattr(update, field) is not None:
                                update_dict[field] = getattr(update, field)
                        if update.metadata:
                            update_dict["metadata"] = update.metadata
                        update_data.append(update_dict)
                    
                    results = await vector_service.update_knowledge_base_content(
                        userId=kb_update.userId,
                        knowledgebaseId=kb_update.knowledgebaseId,
                        updates=update_data,
                        update_type=kb_update.updateType
                    )
                    
                    result = UpdateResponse(
                        success=len(results["failed"]) == 0,
                        message=f"Knowledge base {kb_update.knowledgebaseId}: {len(results['updated'])} updated, {len(results['failed'])} failed",
                        updatedCount=len(results["updated"]),
                        failedCount=len(results["failed"]),
                        errors=results["errors"],
                        updatedIds=results["updated"],
                        timestamp=datetime.now(timezone.utc)
                    )
                    knowledge_base_results.append(result)
                    total_updated += len(results["updated"])
                    total_failed += len(results["failed"])
                    
                except Exception as e:
                    logger.error(f"Error updating knowledge base {kb_update.knowledgebaseId}: {e}")
                    error_result = UpdateResponse(
                        success=False,
                        message=f"Error updating knowledge base {kb_update.knowledgebaseId}",
                        updatedCount=0,
                        failedCount=len(kb_update.updates),
                        errors=[str(e)],
                        timestamp=datetime.now(timezone.utc)
                    )
                    knowledge_base_results.append(error_result)
                    total_failed += len(kb_update.updates)
            
            return BulkUpdateResponse(
                success=total_failed == 0,
                message=f"Bulk update completed: {total_updated} updated, {total_failed} failed",
                conversationResults=conversation_results,
                knowledgeBaseResults=knowledge_base_results,
                totalUpdated=total_updated,
                totalFailed=total_failed,
                timestamp=datetime.now(timezone.utc)
            )
    
    except HTTPException:
        API_ERRORS.labels('/bulk-update').inc()
        raise
    except Exception as e:
        API_ERRORS.labels('/bulk-update').inc()
        logger.error(f"Error in bulk update: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in bulk update: {str(e)}")

@router.delete("/user/{userId}")
async def delete_user_conversations(
    userId: str,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Delete all conversation data for a user from the vector store."""
    try:
        with API_LATENCY.labels('/conversation/delete').time():
            API_REQUESTS.labels('/conversation/delete').inc()

            if not userId:
                raise HTTPException(status_code=400, detail="User ID is required")

            try:
                vector_service = VectorStoreService(config=get_default_vector_config())
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
async def delete_specific_conversation(
    conversationId: str,
    userId: str,
    knowledgebaseId: Optional[str] = None,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Delete a specific conversation from the vector store."""
    try:
        with API_LATENCY.labels('/conversation/delete-specific').time():
            API_REQUESTS.labels('/conversation/delete-specific').inc()

            if not userId:
                raise HTTPException(status_code=400, detail="User ID is required")

            if not conversationId:
                raise HTTPException(status_code=400, detail="Conversation ID is required")

            try:
                vector_service = VectorStoreService(config=get_default_vector_config())
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
async def debug_conversation(
    conversationId: str,
    userId: str,
    knowledgebaseId: Optional[str] = None,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Debug endpoint to check if a conversation exists in the vector store."""
    try:
        API_REQUESTS.labels('/conversation/debug').inc()
        logger.info(f"Debug check for conversation - ConversationId: {conversationId}, UserId: {userId}")

        knowledgebaseId = knowledgebaseId or conversationId
        vector_service = VectorStoreService(config=get_default_vector_config())
        
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