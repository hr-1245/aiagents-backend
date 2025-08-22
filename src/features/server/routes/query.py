from fastapi import APIRouter, HTTPException, Depends, Request
from datetime import datetime, timezone

from ...ai.agents.query_agent import create_query_graph
from ...ai.models.models import QueryResponse, QueryRequest
from ...ai.vector.vector_store import VectorStoreService
from ...ai.vector.multi_kb_service import MultiKnowledgeBaseService
from ...ai.agents.custom_agent_service import custom_agent_service
from ...ai.utils.langgraph_utils import invoke_graph_safely
from ...ai.config import get_llm_config, get_dynamic_llm_config, get_humanlike_behavior_instructions
from ..config import logger, API_REQUESTS, API_ERRORS, API_LATENCY, get_default_vector_config, get_cached_vector_service, get_cached_multi_kb_service, get_cached_custom_agent_service
from langchain_openai import ChatOpenAI

# Create the router for query endpoints
router = APIRouter()

# Import security dependencies - with fallback if not available
try:
    from ..security import authenticate, check_rate_limit
    security_available = True
except ImportError:
    # Fallback dependencies that do nothing
    async def authenticate(request: Request):
        return {"name": "unauthenticated", "permissions": ["read", "write"]}
    
    async def check_rate_limit(request: Request):
        pass
    
    security_available = False

# Import conversation settings store
try:
    from .management import conversation_settings_store
except ImportError:
    conversation_settings_store = {}

@router.post("/query", response_model=QueryResponse)
async def query_conversation(
    request: QueryRequest,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Query conversation history and/or knowledge bases to generate a response using AI agents."""
    try:
        with API_LATENCY.labels('/conversation/query').time():
            API_REQUESTS.labels('/conversation/query').inc()
            
            # Check for saved conversation settings if no agent ID provided
            effective_agent_id = request.aiAgentId
            if not effective_agent_id and request.conversationId:
                settings_key = f"{request.userId}:{request.conversationId}"
                if settings_key in conversation_settings_store:
                    saved_settings = conversation_settings_store[settings_key]
                    effective_agent_id = saved_settings.get("aiAgentId")
                    if effective_agent_id:
                        logger.info(f"Using saved AI agent {effective_agent_id} for conversation {request.conversationId}")
            
            # Determine query mode
            has_conversation = bool(request.conversationId)
            has_knowledge_bases = bool(request.knowledgebaseId or request.additionalKnowledgebaseIds)
            
            if not has_conversation and not has_knowledge_bases:
                raise HTTPException(
                    status_code=400,
                    detail="Either conversationId or knowledgebaseId (or additionalKnowledgebaseIds) must be provided"
                )
            
            logger.info("-" * 40 + " QUERY PROCESSING " + "-" * 40)
            logger.info(f"USER ID      : {request.userId}")
            logger.info(f"CONVERSATION : {request.conversationId}")
            logger.info(f"QUERY        : '{request.query}'")
            if effective_agent_id:
                logger.info(f"AI AGENT     : {effective_agent_id} {'(from settings)' if effective_agent_id != request.aiAgentId else '(from request)'}")
            if request.additionalKnowledgebaseIds:
                logger.info(f"EXTRA KBs    : {request.additionalKnowledgebaseIds}")
            logger.info("-" * 90)

            if has_conversation:
                # CONVERSATION + KB MODE: Conversation is primary, custom KBs enhance
                primary_kb_id = request.knowledgebaseId or request.conversationId
                additional_kb_ids = request.additionalKnowledgebaseIds or []

                # Get context from conversation (primary) + custom KBs (enhancement)
                logger.info("STEP 1: Retrieving context from conversation + knowledge bases")
                retrieved_context = await _get_enhanced_context(
                    user_id=request.userId,
                    query=request.query,
                    primary_kb_id=primary_kb_id,
                    conversation_id=request.conversationId,
                    additional_kb_ids=additional_kb_ids,
                    limit=request.limit or 5
                )
                # Safe context logging (truncate to avoid Unicode issues)
                context_preview = retrieved_context[:200] if retrieved_context else "No context"
                logger.info(f"         Context retrieved: {len(retrieved_context)} chars")
                if len(context_preview) > 0:
                    # Remove problematic Unicode characters for logging
                    safe_context = context_preview.encode('ascii', 'ignore').decode('ascii')
                    logger.debug(f"         Context preview: {safe_context}...")

            else:
                # KB-ONLY MODE: Knowledge bases only, no conversation
                primary_kb_id = request.knowledgebaseId
                additional_kb_ids = request.additionalKnowledgebaseIds or []

                logger.info("STEP 1: Retrieving context from knowledge bases only")
                retrieved_context = await _get_kb_only_context(
                    user_id=request.userId,
                    query=request.query,
                    primary_kb_id=primary_kb_id,
                    additional_kb_ids=additional_kb_ids,
                    limit=request.limit or 5
                )
                logger.info(f"         KB context retrieved: {len(retrieved_context)} chars")

            # Try custom agent first if specified (use effective_agent_id)
            if effective_agent_id or request.agentInfo:
                logger.info(f"STEP 2: Attempting to use custom agent")
                try:
                    # Try to get agent from service first
                    agent = None
                    cached_agent_service = None
                    if effective_agent_id:
                        cached_agent_service = await get_cached_custom_agent_service()
                        agent = await cached_agent_service.get_agent(effective_agent_id, request.userId)
                    
                    if agent and agent.agentType == "query" and agent.isActive:
                        logger.info(f"         Custom agent found: {agent.name}")
                        context = {
                            "context": retrieved_context,
                            "query": request.query,
                            "mode": "conversation+kb" if has_conversation else "kb_only"
                        }
                        
                        response_text = await cached_agent_service.execute_agent(
                            effective_agent_id, request.userId, context,
                            temperature=request.temperature,
                            model=request.model,
                            humanlike_behavior=request.humanlikeBehavior
                        )
                        logger.info(f"         Custom agent response: {len(response_text)} chars")
                        
                        # Return basic suggestions since we removed hardcoded defaults
                        suggestions = [
                            "What specific information would be most helpful?",
                            "How can I better assist you with this matter?",
                            "Are there any other questions I can help answer?"
                        ]
                        
                        logger.info("         Custom agent executed successfully")
                        logger.info("=" * 90)
                        return QueryResponse(
                            messages=[],  # Not needed for agent responses
                            total=0,
                            query=request.query,
                            answer=response_text,
                            suggestions=suggestions,
                            timestamp=datetime.now(timezone.utc)
                        )
                    
                    # If no stored agent, use frontend agentInfo
                    elif request.agentInfo:
                        logger.info(f"         Using frontend agent info: {request.agentInfo.name}")
                        
                        # Create custom prompt using agent info
                        custom_prompt = f"{request.agentInfo.prompt}\n\nCONVERSATION CONTEXT:\n{retrieved_context}\n\nCUSTOMER'S QUESTION:\n\"{request.query}\"\n\nCOMPREHENSIVE RESPONSE INSTRUCTIONS:\n- Leverage ALL available sources: conversation history, multiple knowledge bases, document repositories, and contextual information\n- SYNTHESIZE information across multiple sources to provide comprehensive, authoritative answers\n- Use conversation context to understand customer intent, then enhance with relevant knowledge base content\n- For any topic (products, services, procedures, troubleshooting), cross-reference multiple knowledge sources\n- When providing information, cite or reference the source type when helpful (conversation history, knowledge base, documentation)\n- If conversation context is limited, proactively supplement with relevant knowledge base information\n- Provide multi-layered responses that combine conversational context with knowledge base insights\n- Always aim to exceed customer expectations by providing more comprehensive information than requested\n- Suggest related topics or information from knowledge bases that might be valuable to the customer\n- Never give partial answers when comprehensive information is available across multiple sources\n- Stay in character as {request.agentInfo.name} while leveraging all available information sources."
                        
                        # Add human-like behavior instructions if requested
                        if request.humanlikeBehavior:
                            custom_prompt += f"\n\n{get_humanlike_behavior_instructions()}"
                        
                        # Execute with direct LLM call using dynamic config
                        llm_config = get_dynamic_llm_config(
                            temperature=request.temperature,
                            model=request.model
                        )
                        llm = ChatOpenAI(**llm_config)
                        result = await llm.ainvoke(custom_prompt)
                        response_text = result.content or ""
                        
                        # Return basic suggestions since we removed hardcoded defaults
                        suggestions = [
                            "What specific information would be most helpful?",
                            "How can I better assist you with this matter?",
                            "Are there any other questions I can help answer?"
                        ]
                        
                        logger.info(f"         Frontend agent executed successfully: {len(response_text)} chars")
                        logger.info("=" * 90)
                        return QueryResponse(
                            messages=[],
                            total=0,
                            query=request.query,
                            answer=response_text,
                            suggestions=suggestions,
                            timestamp=datetime.now(timezone.utc)
                        )
                    else:
                        logger.warning(f"         Custom agent not found or inactive: {effective_agent_id}")
                        
                except Exception as e:
                    logger.warning(f"         Custom agent failed, using default: {e}")

            # Use default query agent
            logger.info("STEP 2: Using default query agent")
            response_text, suggestions = await _use_default_query_agent(
                query=request.query,
                context=retrieved_context,
                user_id=request.userId,
                conversation_id=request.conversationId,
                primary_kb_id=request.knowledgebaseId or request.conversationId,
                mode="conversation+kb" if has_conversation else "kb_only"
            )

            logger.info("STEP 3: Query processing complete")
            logger.info(f"         Response length: {len(response_text)} chars")
            logger.info(f"         Generated {len(suggestions)} suggestions")
            logger.info("=" * 90)
            
            return QueryResponse(
                messages=[],  # Simplified - context is handled internally
                total=0,
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
            detail=f"Error processing query: {str(e)}"
        )


async def _get_enhanced_context(
    user_id: str,
    query: str,
    primary_kb_id: str,
    conversation_id: str,
    additional_kb_ids: list,
    limit: int = 5
) -> str:
    """Get context from conversation (primary) + custom knowledge bases (enhancement)."""
    try:
        if additional_kb_ids:
            # Use cached multi-KB service for enhanced context
            multi_kb_service = await get_cached_multi_kb_service(user_id)
            all_kb_ids = [primary_kb_id] + additional_kb_ids
            
            results = await multi_kb_service.query_multiple_knowledge_bases(
                user_id=user_id,
                query=query,
                knowledge_base_ids=all_kb_ids,
                k=limit,
                primary_kb_id=primary_kb_id
            )
            
            # Format results with source priority (conversation first)
            formatted_context = []
            
            # Prioritize conversation context
            for result in results:
                if result.get('knowledgebaseId') == primary_kb_id:
                    role = result.get('role', 'unknown')
                    body = result.get('body', '')
                    formatted_context.append(f"[Conversation - {role}] {body}")
            
            # Add custom KB context as enhancement
            for result in results:
                if result.get('knowledgebaseId') != primary_kb_id:
                    kb_id = result.get('knowledgebaseId', 'unknown')
                    body = result.get('body', '')
                    formatted_context.append(f"[Knowledge Base - {kb_id}] {body}")
            
            return "\n".join(formatted_context) if formatted_context else "No relevant context found."
        
        else:
            # Use cached single KB service for conversation only
            vector_service = await get_cached_vector_service(user_id)
            
            retrieved_messages = await vector_service.query_chat_history(
                userId=user_id,
                query=query,
                metadata_filter={
                    "conversationId": conversation_id,
                    "knowledgebaseId": primary_kb_id
                },
                k=limit,
                knowledgebaseId=primary_kb_id
            )
            
            if retrieved_messages:
                formatted_messages = []
                for msg in retrieved_messages:
                    role = msg.get('role', 'unknown')
                    body = msg.get('body', '')
                    formatted_messages.append(f"[{role}] {body}")
                return "\n".join(formatted_messages)
            else:
                return "No relevant conversation history found."
                
    except Exception as e:
        logger.error(f"Error getting enhanced context: {e}")
        return "Error retrieving context."


async def _get_kb_only_context(
    user_id: str,
    query: str,
    primary_kb_id: str,
    additional_kb_ids: list,
    limit: int = 5
) -> str:
    """Get context from knowledge bases only (no conversation)."""
    try:
        if not primary_kb_id:
            return "No knowledge base specified."
        
        # Use cached multi-KB service for querying multiple knowledge bases
        multi_kb_service = await get_cached_multi_kb_service(user_id)
        
        results = await multi_kb_service.query_multiple_knowledge_bases(
            user_id=user_id,
            query=query,
            knowledge_base_ids=[primary_kb_id] + additional_kb_ids,
            k=limit
        )
        
        # Format results with knowledge base labels
        formatted_context = []
        for result in results:
            kb_id = result.get('knowledgebaseId', 'unknown')
            body = result.get('body', '')
            source = result.get('source', 'document')
            formatted_context.append(f"[Knowledge Base - {kb_id}] ({source}) {body}")
        
        return "\n".join(formatted_context) if formatted_context else "No relevant knowledge base content found."
        
    except Exception as e:
        logger.error(f"Error getting knowledge base only context: {e}")
        return "Error retrieving knowledge base context."


async def _use_default_query_agent(
    query: str,
    context: str,
    user_id: str,
    conversation_id: str,
    primary_kb_id: str,
    mode: str
) -> tuple[str, list]:
    """Use the default query agent to generate response and suggestions."""
    try:
        # If no custom agent specified, return error - don't use hardcoded defaults
        logger.warning("No custom agent specified for query - using basic response")
        
        # Use basic query processing without hardcoded agent IDs
        graph = create_query_graph()
        result = await invoke_graph_safely(
            graph,
            {
                "query": query,
                "context": context,
                "mode": mode
            }
        )
        
        response_text = result.get("answer", "I'm sorry, I couldn't process your query at this time.")
        
        # Return basic suggestions without hardcoded agent IDs
        suggestions = [
            "What specific information would be most helpful?",
            "How can I better assist you with this matter?",
            "Are there any other questions I can help answer?"
        ]
        
        return response_text, suggestions
        
    except Exception as e:
        logger.error(f"Error in query processing: {e}")
        return "I'm sorry, I encountered an error while processing your query.", [] 