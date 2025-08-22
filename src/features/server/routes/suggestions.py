from fastapi import APIRouter, HTTPException, Depends
from langchain_openai import ChatOpenAI
from datetime import datetime

from ...ai.utils.conversation_utils import (
    format_conversation_context, build_suggestion_prompt,
    build_response_prompt, extract_numbered_items,
    calculate_autopilot_confidence,
    FALLBACK_SUGGESTIONS, FALLBACK_RESPONSES
)
from ...ai.models.models import (
    SuggestionRequest, SuggestionResponse, ResponseSuggestionRequest, ResponseSuggestionResponse,
    EnhancedSuggestionRequest, EnhancedResponseSuggestionRequest
)
from ...ai.vector.vector_store import VectorStoreService
from ...ai.vector.multi_kb_service import MultiKnowledgeBaseService
from ...ai.agents.custom_agent_service import custom_agent_service
from ...ai.config import get_llm_config, get_dynamic_llm_config, get_humanlike_behavior_instructions
from ...ai.utils.langgraph_utils import invoke_graph_safely
from ..config import logger, get_default_vector_config, get_cached_custom_agent_service, get_cached_multi_kb_service
from ..security import authenticate, check_rate_limit

# Import enhanced agents
try:
    from ...ai.agents.enhanced_suggestion_agent import create_enhanced_suggestion_graph
    from ...ai.agents.response_suggestions_agent import create_response_suggestions_graph
    ENHANCED_AGENTS_AVAILABLE = True
    logger.info("Enhanced agents loaded successfully")
except ImportError as e:
    logger.warning(f"Enhanced agents not available: {e}")
    ENHANCED_AGENTS_AVAILABLE = False

# Import conversation settings store
try:
    from .management import conversation_settings_store
except ImportError:
    conversation_settings_store = {}

# Create the router for suggestion endpoints
router = APIRouter()

def _build_autopilot_prompt(
    last_customer_message: str,
    conversation_context: str,
    recent_messages: list,
    system_prompt: str = None,
    agent_name: str = None,
    customer_info: dict = None,
    additional_context: str = ""
) -> str:
    """Build enhanced prompt for autopilot mode."""
    
    # Base autopilot instructions
    base_instructions = """You are a professional customer service representative generating a single, high-quality response for autopilot mode.

AUTOPILOT REQUIREMENTS:
- Generate ONE ready-to-send response (50-200 words)
- Be professional, helpful, and empathetic
- Use conversation context to personalize the response
- Address the customer's specific concern directly
- Offer concrete next steps or assistance
- Sound natural and conversational
- NO markdown formatting - plain text only
- Be confident but not overconfident

"""
    
    # Add agent-specific instructions
    if system_prompt:
        base_instructions += f"\nAGENT INSTRUCTIONS:\n{system_prompt}\n"
    
    if agent_name:
        base_instructions += f"\nYou are {agent_name}, representing our customer service team.\n"
    
    # Build context section
    context_section = ""
    if conversation_context:
        context_section += f"\nKNOWLEDGE BASE CONTEXT:\n{conversation_context}\n"
    
    if recent_messages:
        context_section += "\nRECENT CONVERSATION:\n"
        for msg in recent_messages[-5:]:  # Last 5 messages
            role = msg.get('role', 'unknown')
            content = msg.get('body', msg.get('content', ''))
            if content:
                context_section += f"[{role}]: {content}\n"
    
    if customer_info:
        context_section += f"\nCUSTOMER INFO:\n{customer_info}\n"
    
    if additional_context:
        context_section += f"\nADDITIONAL CONTEXT:\n{additional_context}\n"
    
    # Customer's current message
    customer_section = f"\nCUSTOMER'S MESSAGE:\n\"{last_customer_message}\"\n"
    
    # Final instruction
    final_instruction = """
Generate a single, comprehensive response that addresses the customer's message using all available context. The response should be ready to send directly to the customer and demonstrate understanding of their situation while providing helpful assistance.

RESPONSE:"""
    
    return base_instructions + context_section + customer_section + final_instruction

def _calculate_enhanced_autopilot_confidence(
    context: str,
    last_message: str,
    response: str,
    recent_messages: list = None
) -> float:
    """Calculate enhanced confidence score for autopilot responses."""
    
    confidence = 0.5  # Base confidence
    
    # Context availability (0.2 weight)
    if context and len(context.strip()) > 50:
        confidence += 0.15
    elif context and len(context.strip()) > 20:
        confidence += 0.1
    
    # Recent messages context (0.15 weight)
    if recent_messages and len(recent_messages) > 0:
        confidence += 0.1
        if len(recent_messages) >= 3:
            confidence += 0.05
    
    # Response quality indicators (0.3 weight)
    if response and len(response.strip()) > 30:
        confidence += 0.1
        
        # Check for helpful indicators
        helpful_indicators = [
            "help", "assist", "support", "resolve", "solution", 
            "understand", "sorry", "apologize", "thank", "welcome"
        ]
        response_lower = response.lower()
        helpful_count = sum(1 for indicator in helpful_indicators if indicator in response_lower)
        confidence += min(helpful_count * 0.03, 0.15)
    
    # Message clarity and relevance (0.2 weight)
    if last_message and len(last_message.strip()) > 5:
        confidence += 0.1
        
        # Check if response seems relevant to the message
        message_words = set(last_message.lower().split())
        response_words = set(response.lower().split())
        common_words = message_words.intersection(response_words)
        if len(common_words) > 0:
            confidence += min(len(common_words) * 0.02, 0.1)
    
    # Avoid overconfidence for very short responses
    if len(response.strip()) < 20:
        confidence *= 0.7
    
    # Avoid overconfidence for generic responses
    generic_phrases = ["i'd be happy to help", "please provide more details", "how can i help"]
    if any(phrase in response.lower() for phrase in generic_phrases):
        confidence *= 0.8
    
    return min(max(confidence, 0.0), 1.0)  # Clamp between 0 and 1

async def _get_context_data(
    user_id: str,
    conversation_id: str,
    knowledgebase_id: str,
    additional_kb_ids: list,
    recent_messages: list,
    query: str = None
):
    """Helper function to get context data from single or multiple knowledge bases."""
    
    # Initialize services
    vector_config = get_default_vector_config()
    vector_service = VectorStoreService(config=vector_config)
    multi_kb_service = MultiKnowledgeBaseService(vector_config)
    
    messages = recent_messages or []
    vector_context = ""
    
    # Determine mode
    has_conversation = bool(conversation_id)
    
    # Get messages if not provided and we have a conversation
    if not messages and has_conversation:
        try:
            if additional_kb_ids:
                # Multi-KB mode
                messages = await multi_kb_service.get_conversation_from_multiple_kbs(
                    user_id, conversation_id, knowledgebase_id, additional_kb_ids
                )
            else:
                # Single KB mode
                results = await vector_service.get_conversation_by_id(
                    userId=user_id,
                    conversationId=conversation_id,
                    knowledgebaseId=knowledgebase_id
                )
                results.sort(key=lambda r: r.get("dateAdded", ""))
                messages = results[-5:]
        except Exception:
            messages = []
    
    # Get vector context
    if query:
        try:
            if has_conversation and additional_kb_ids:
                # Conversation + KB mode: Multi-KB vector query
                all_kb_ids = [knowledgebase_id] + additional_kb_ids
                vector_results = await multi_kb_service.query_multiple_knowledge_bases(
                    user_id=user_id,
                    query=query,
                    knowledge_base_ids=all_kb_ids,
                    k=5,
                    primary_kb_id=knowledgebase_id
                )
                vector_context = multi_kb_service.merge_contexts(vector_results)
            elif has_conversation:
                # Conversation only mode: Single KB vector query
                vector_results = await vector_service.query_chat_history(
                    userId=user_id,
                    query=query,
                    metadata_filter={
                        "conversationId": conversation_id,
                        "knowledgebaseId": knowledgebase_id
                    },
                    k=5,
                    knowledgebaseId=knowledgebase_id
                )
                if vector_results:
                    vector_context = format_conversation_context(vector_results)
            else:
                # Knowledge base only mode
                kb_ids = []
                if knowledgebase_id:
                    kb_ids.append(knowledgebase_id)
                if additional_kb_ids:
                    kb_ids.extend(additional_kb_ids)
                
                if kb_ids:
                    vector_results = await multi_kb_service.query_multiple_knowledge_bases(
                        user_id=user_id,
                        query=query,
                        knowledge_base_ids=kb_ids,
                        k=5
                    )
                    # Format KB-only results
                    formatted_context = []
                    for result in vector_results:
                        kb_id = result.get('knowledgebaseId', 'unknown')
                        body = result.get('body', '')
                        source = result.get('source', 'document')
                        formatted_context.append(f"[Knowledge Base - {kb_id}] ({source}) {body}")
                    vector_context = "\n".join(formatted_context)
        except Exception:
            pass
    
    return messages, vector_context

async def _get_context_for_suggestions(
    request: EnhancedSuggestionRequest,
    has_conversation: bool,
    multi_kb_service: MultiKnowledgeBaseService
) -> tuple[list, str]:
    """Helper to get context data for suggestion generation."""
    
    # Get messages
    messages = request.recentMessages or []
    
    # Get vector context
    vector_context = ""
    
    # Get the vector service for single KB operations
    vector_config = get_default_vector_config()
    vector_service = VectorStoreService(config=vector_config)
    
    if request.conversationId:
        if request.knowledgebaseId or request.additionalKnowledgebaseIds:
            # Conversation + KB mode: Multi-KB vector query
            all_kb_ids = []
            if request.knowledgebaseId:
                all_kb_ids.append(request.knowledgebaseId)
            if request.additionalKnowledgebaseIds:
                all_kb_ids.extend(request.additionalKnowledgebaseIds)
                
            vector_results = await multi_kb_service.query_multiple_knowledge_bases(
                user_id=request.userId,
                query=request.context or request.query or "context for suggestions",
                knowledge_base_ids=all_kb_ids,
                k=5,
                primary_kb_id=request.knowledgebaseId
            )
            vector_context = multi_kb_service.merge_contexts(vector_results)
        else:
            # Conversation only mode: Single KB vector query
            vector_results = await vector_service.query_chat_history(
                userId=request.userId,
                query=request.context or request.query or "context for suggestions",
                metadata_filter={
                    "conversationId": request.conversationId,
                    "knowledgebaseId": request.knowledgebaseId or (request.conversationId if has_conversation else None)
                },
                k=5,
                knowledgebaseId=request.knowledgebaseId or (request.conversationId if has_conversation else None)
            )
            if vector_results:
                vector_context = format_conversation_context(vector_results)
    elif request.knowledgebaseId or request.additionalKnowledgebaseIds:
        # KB-only mode: Multi-KB vector query
        kb_ids = []
        if request.knowledgebaseId:
            kb_ids.append(request.knowledgebaseId)
        if request.additionalKnowledgebaseIds:
            kb_ids.extend(request.additionalKnowledgebaseIds)
        
        if kb_ids:
            vector_results = await multi_kb_service.query_multiple_knowledge_bases(
                user_id=request.userId,
                query=request.context or request.query or "context for suggestions",
                knowledge_base_ids=kb_ids,
                k=5
            )
            # Format KB-only results
            formatted_context = []
            for result in vector_results:
                kb_id = result.get('knowledgebaseId', 'unknown')
                body = result.get('body', '')
                source = result.get('source', 'document')
                formatted_context.append(f"[Knowledge Base - {kb_id}] ({source}) {body}")
            vector_context = "\n".join(formatted_context)
    
    return messages, vector_context

@router.post("/suggestions", response_model=SuggestionResponse)
async def generate_suggestions(
    request: SuggestionRequest,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Generate follow-up messages for a conversation or knowledge base queries using default or custom agents."""
    try:
        # Check for saved conversation settings if no agent ID provided
        effective_agent_id = request.aiAgentId
        if not effective_agent_id and request.conversationId:
            settings_key = f"{request.userId}:{request.conversationId}"
            if settings_key in conversation_settings_store:
                saved_settings = conversation_settings_store[settings_key]
                effective_agent_id = saved_settings.get("aiAgentId")
                if effective_agent_id:
                    logger.info(f"Using saved AI agent {effective_agent_id} for suggestions in conversation {request.conversationId}")
        
        logger.info(f"SUGGESTIONS REQUEST RECEIVED", extra={
            "userId": request.userId,
            "conversationId": request.conversationId,
            "query": request.query,
            "knowledgebaseId": request.knowledgebaseId,
            "additionalKBs": request.additionalKnowledgebaseIds,
            "aiAgentId": effective_agent_id,
            "limit": request.limit,
            "timestamp": datetime.now().isoformat()
        })
        
        # Validate request
        has_conversation = bool(request.conversationId)
        has_knowledge_bases = bool(request.knowledgebaseId or request.additionalKnowledgebaseIds)
        
        logger.debug(f"SUGGESTIONS - Request validation - has_conversation: {has_conversation}, has_knowledge_bases: {has_knowledge_bases}")
        
        if not has_conversation and not has_knowledge_bases:
            logger.warning(f"SUGGESTIONS - Invalid request - missing conversation or knowledge base")
            raise HTTPException(
                status_code=400,
                detail="Either conversationId or knowledgebaseId (or additionalKnowledgebaseIds) must be provided"
            )
        
        # Convert to enhanced request for internal processing (use effective_agent_id)
        enhanced_request = EnhancedSuggestionRequest(
            userId=request.userId,
            conversationId=request.conversationId,
            query=request.query,
            context=request.context,
            knowledgebaseId=request.knowledgebaseId or (request.conversationId if has_conversation else None),
            additionalKnowledgebaseIds=request.additionalKnowledgebaseIds,
            aiAgentId=effective_agent_id,  # Use effective agent ID from settings
            limit=request.limit,
            recentMessages=request.recentMessages
        )
        
        return await _generate_suggestions_internal(enhanced_request)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Suggestion generation failed: {str(e)}")

@router.post("/suggestions/enhanced", response_model=SuggestionResponse)
async def generate_enhanced_suggestions(
    request: EnhancedSuggestionRequest,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Generate follow-up messages with custom agent and multi-KB support."""
    try:
        # Validate request
        has_conversation = bool(request.conversationId)
        has_knowledge_bases = bool(request.knowledgebaseId or request.additionalKnowledgebaseIds)
        
        if not has_conversation and not has_knowledge_bases:
            raise HTTPException(
                status_code=400,
                detail="Either conversationId or knowledgebaseId (or additionalKnowledgebaseIds) must be provided"
            )
        
        return await _generate_suggestions_internal(request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced suggestion generation failed: {str(e)}")

async def _generate_suggestions_internal(request: EnhancedSuggestionRequest):
    """Internal function to handle suggestion generation."""
    
    # Determine mode
    has_conversation = bool(request.conversationId)
    mode = "conversation+kb" if has_conversation else "kb_only"
    
    # Get services
    cached_agent_service = await get_cached_custom_agent_service()
    cached_multi_kb_service = await get_cached_multi_kb_service()
    
    # Initialize context gathering
    messages, vector_context = await _get_context_for_suggestions(
        request, has_conversation, cached_multi_kb_service
    )
    
    limit = request.limit or 3
    
    logger.info(f"SUGGESTIONS MODE: {mode}")
    logger.info(f"Vector context length: {len(vector_context or '')} characters")
    logger.info(f"Message count: {len(messages)}")
    
    # Try custom agent first if specified
    if request.aiAgentId:
        try:
            logger.info(f"Attempting to use custom agent: {request.aiAgentId}")
            agent = await cached_agent_service.get_agent(request.aiAgentId, request.userId)
            
            if agent and agent.isActive:
                logger.info(f"Custom agent found: {agent.name} (type: {agent.agentType})")
                
                # Get the last customer message for context
                last_customer_message = ""
                if messages:
                    for msg in reversed(messages):
                        if msg.get("direction") == "inbound" or msg.get("role") == "customer":
                            last_customer_message = msg.get("body", "")
                            break
                
                # Use the agent's prompt for suggestion generation
                agent_prompt = f"""{agent.prompt}

CONVERSATION CONTEXT:
{vector_context or "No previous context available"}

CUSTOMER'S LAST MESSAGE:
"{last_customer_message or request.query or 'Generate helpful suggestions'}"

TASK: Generate {limit} helpful, professional response suggestions that this customer service agent would provide.

REQUIREMENTS:
- Each suggestion should be numbered (1., 2., 3., etc.)
- Suggestions should be conversational and ready to send
- Use the conversation context to make suggestions relevant and specific
- Stay in character as {agent.name}
- Follow the agent instructions above
- Be helpful, professional, and empathetic
- Address the customer's specific needs or questions
- Offer concrete next steps or assistance

Generate {limit} numbered suggestions:"""
                
                # Add human-like behavior instructions if requested
                if request.humanlikeBehavior:
                    agent_prompt += f"\n\n{get_humanlike_behavior_instructions()}"
                
                # Execute with direct LLM call using agent's prompt
                llm_config = get_dynamic_llm_config(
                    temperature=request.temperature,
                    model=request.model
                )
                llm = ChatOpenAI(**llm_config)
                result = await llm.ainvoke(agent_prompt)
                suggestions_text = result.content or ""
                
                suggestions = extract_numbered_items(suggestions_text)
                
                if suggestions and len(suggestions) >= min(2, limit):
                    logger.info(f"Custom agent prompt generated {len(suggestions)} suggestions")
                    return SuggestionResponse(
                        suggestions=suggestions[:limit],
                        total=len(suggestions[:limit]),
                        conversationId=request.conversationId or "kb_query"
                    )
                else:
                    logger.warning(f"Custom agent {request.aiAgentId} prompt generated insufficient suggestions")
            else:
                logger.warning(f"Custom agent {request.aiAgentId} not found or not active")
        except Exception as e:
            logger.warning(f"Custom agent {request.aiAgentId} failed: {e}")

    # If no custom agent found or failed, use agentInfo from frontend (like query function)
    if hasattr(request, 'agentInfo') and request.agentInfo:
        try:
            logger.info(f"Using frontend agent info: {request.agentInfo.name}")
            
            # Get the last customer message for context
            last_customer_message = ""
            if messages:
                for msg in reversed(messages):
                    if msg.get("direction") == "inbound" or msg.get("role") == "customer":
                        last_customer_message = msg.get("body", "")
                        break
            
            # Create custom prompt using agent info
            custom_prompt = f"""{request.agentInfo.prompt}

CONVERSATION CONTEXT:
{vector_context or "No previous context available"}

CUSTOMER'S LAST MESSAGE:
"{last_customer_message or request.query or 'Generate helpful suggestions'}"

TASK: Generate {limit} helpful, professional response suggestions that this customer service agent would provide. 

REQUIREMENTS:
- Each suggestion should be numbered (1., 2., 3., etc.)
- Suggestions should be conversational and ready to send
- Use the conversation context to make suggestions relevant
- Stay in character as {request.agentInfo.name}
- Be helpful, professional, and empathetic
- Address the customer's specific needs or questions

Generate {limit} numbered suggestions:"""
            
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
            suggestions_text = result.content or ""
            
            suggestions = extract_numbered_items(suggestions_text)
            
            if suggestions:
                logger.info(f"Frontend agent info generated {len(suggestions)} suggestions")
                return SuggestionResponse(
                    suggestions=suggestions[:limit],
                    total=len(suggestions[:limit]),
                    conversationId=request.conversationId or "kb_query"
                )
        except Exception as e:
            logger.warning(f"Frontend agent info failed: {e}")

    # No agent specified - use intelligent AI-generated suggestions (like query function)
    if not request.aiAgentId:
        logger.info("No agent ID provided - generating intelligent AI suggestions")
        
        try:
            # Get the last customer message for context
            last_customer_message = ""
            if messages:
                for msg in reversed(messages):
                    if msg.get("direction") == "inbound" or msg.get("role") == "customer":
                        last_customer_message = msg.get("body", "")
                        break
            
            # Use context-aware prompt for intelligent suggestions
            intelligent_prompt = f"""You are a professional customer service AI generating helpful response suggestions.

CONVERSATION CONTEXT:
{vector_context or "No previous context available"}

CUSTOMER'S LAST MESSAGE:
"{last_customer_message or request.query or 'Customer needs assistance'}"

TASK: Generate {limit} helpful, professional response suggestions that a customer service representative would use.

REQUIREMENTS:
- Each suggestion should be numbered (1., 2., 3., etc.)
- Suggestions should be conversational and ready to send
- Use the conversation context to make suggestions relevant and specific
- Be helpful, professional, and empathetic
- Address the customer's specific needs or questions
- Offer concrete next steps or assistance

Generate {limit} numbered suggestions:"""
            
            # Execute with default config
            llm_config = get_dynamic_llm_config(
                temperature=request.temperature or 0.7,
                model=request.model
            )
            llm = ChatOpenAI(**llm_config)
            result = await llm.ainvoke(intelligent_prompt)
            suggestions_text = result.content or ""
            
            # Debug logging
            logger.info(f"AI Raw Response Length: {len(suggestions_text)}")
            logger.debug(f"AI Raw Response (first 200 chars): {suggestions_text[:200]}...")
            
            suggestions = extract_numbered_items(suggestions_text)
            
            # More debugging
            logger.info(f"Extracted {len(suggestions)} suggestions from AI response")
            if suggestions:
                for i, s in enumerate(suggestions, 1):
                    logger.debug(f"Suggestion {i}: {s[:100]}...")
            
            if suggestions:
                logger.info(f"Intelligent AI generated {len(suggestions)} suggestions")
                return SuggestionResponse(
                    suggestions=suggestions[:limit],
                    total=len(suggestions[:limit]),
                    conversationId=request.conversationId or "kb_query"
                )
            else:
                logger.warning("AI generated response but no suggestions extracted")
                logger.debug(f"Full AI response for debugging: {suggestions_text}")
        except Exception as e:
            logger.warning(f"Intelligent AI suggestions failed: {e}")
            logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")
    
    # Final fallback to enhanced agents or basic suggestions
    if ENHANCED_AGENTS_AVAILABLE and has_conversation:
        try:
            # Find last customer message (only for conversation mode)
            last_customer_message = None
            if messages:
                for msg in reversed(messages):
                    if msg.get("direction") == "inbound" or msg.get("role") == "customer":
                        last_customer_message = msg.get("body", "")
                        break

            if last_customer_message:
                graph = create_enhanced_suggestion_graph()
                result = await invoke_graph_safely(
                    graph,
                    {
                        "last_customer_message": last_customer_message,
                        "conversation_context": vector_context or format_conversation_context(messages),
                        "mode": mode
                    }
                )
                
                suggestions = result.get("suggestions", [])
                
                if suggestions and len(suggestions) >= min(2, limit):
                    logger.info(f"Enhanced agent generated {len(suggestions)} suggestions")
                    return SuggestionResponse(
                        suggestions=suggestions[:limit],
                        total=len(suggestions[:limit]),
                        conversationId=request.conversationId or "kb_query"
                    )
        except Exception as e:
            logger.warning(f"Enhanced suggestion agent failed: {e}")

    # Ultimate fallback - return contextual basic suggestions
    logger.info("Using basic fallback suggestions")
    
    # Try to make fallback suggestions more contextual
    if vector_context or messages:
        fallback_suggestions = [
            "How can I help you with this specific issue?",
            "What additional information would be helpful?",
            "Is there anything else you'd like to know about this topic?"
        ]
    else:
        fallback_suggestions = [
            "What specific information would be most helpful?", 
            "How can I better assist you with this matter?",
            "Are there any other questions I can help answer?"
        ]
    
    return SuggestionResponse(
        suggestions=fallback_suggestions[:limit],
        total=len(fallback_suggestions[:limit]),
        conversationId=request.conversationId or "kb_query"
    )

@router.post("/response-suggestions", response_model=ResponseSuggestionResponse)
async def generate_response_suggestions(
    request: ResponseSuggestionRequest,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Generate response suggestions for customer messages in conversations or knowledge base queries."""
    try:
        # Check for saved conversation settings if no agent ID provided
        effective_agent_id = request.aiAgentId
        if not effective_agent_id and request.conversationId:
            settings_key = f"{request.userId}:{request.conversationId}"
            if settings_key in conversation_settings_store:
                saved_settings = conversation_settings_store[settings_key]
                effective_agent_id = saved_settings.get("aiAgentId")
                if effective_agent_id:
                    logger.info(f"Using saved AI agent {effective_agent_id} for response suggestions in conversation {request.conversationId}")
        
        # Validate request
        has_conversation = bool(request.conversationId)
        has_knowledge_bases = bool(request.knowledgebaseId or request.additionalKnowledgebaseIds)
        
        if not has_conversation and not has_knowledge_bases:
            raise HTTPException(
                status_code=400,
                detail="Either conversationId or knowledgebaseId (or additionalKnowledgebaseIds) must be provided"
            )
        
        # Convert to enhanced request for internal processing (use effective_agent_id)
        enhanced_request = EnhancedResponseSuggestionRequest(
            userId=request.userId,
            conversationId=request.conversationId,
            knowledgebaseId=request.knowledgebaseId or (request.conversationId if has_conversation else None),
            additionalKnowledgebaseIds=request.additionalKnowledgebaseIds,
            aiAgentId=effective_agent_id,  # Use effective agent ID from settings
            context=request.context,
            lastCustomerMessage=request.lastCustomerMessage,
            recentMessages=request.recentMessages,
            autopilot=request.autopilot
        )
        
        return await _generate_response_suggestions_internal(enhanced_request)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Response suggestion generation failed: {str(e)}")

@router.post("/response-suggestions/enhanced", response_model=ResponseSuggestionResponse)
async def generate_enhanced_response_suggestions(
    request: EnhancedResponseSuggestionRequest,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Generate response suggestions with custom agent and multi-KB support."""
    try:
        # Validate request
        has_conversation = bool(request.conversationId)
        has_knowledge_bases = bool(request.knowledgebaseId or request.additionalKnowledgebaseIds)
        
        if not has_conversation and not has_knowledge_bases:
            raise HTTPException(
                status_code=400,
                detail="Either conversationId or knowledgebaseId (or additionalKnowledgebaseIds) must be provided"
            )
        
        return await _generate_response_suggestions_internal(request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced response suggestion generation failed: {str(e)}")

async def _generate_response_suggestions_internal(request: EnhancedResponseSuggestionRequest):
    """Internal function to handle response suggestion generation."""
    
    # Determine mode
    has_conversation = bool(request.conversationId)
    mode = "conversation+kb" if has_conversation else "kb_only"
    
    # Get context data
    messages, vector_context = await _get_context_data(
        user_id=request.userId,
        conversation_id=request.conversationId,
        knowledgebase_id=request.knowledgebaseId or (request.conversationId if has_conversation else None),
        additional_kb_ids=request.additionalKnowledgebaseIds or [],
        recent_messages=request.recentMessages,
        query=request.context or request.lastCustomerMessage or "response context"
    )

    # Get last customer message
    last_customer_message = request.lastCustomerMessage
    if not last_customer_message and has_conversation and messages:
        # Find last customer message from conversation
        for msg in reversed(messages):
            if msg.get("direction") == "inbound" or msg.get("role") == "customer":
                last_customer_message = msg.get("body", "")
                break
    
    # For KB-only mode, use context as the "customer message"
    if not has_conversation and not last_customer_message:
        last_customer_message = request.context or "How can I help you with this knowledge base?"

    if not last_customer_message:
        return ResponseSuggestionResponse(
            response_suggestion="I'd be happy to help! Could you please provide more details about what you need?",
            response_suggestions=["I'd be happy to help! Could you please provide more details about what you need?"],
            total=1,
            conversationId=request.conversationId or "kb_query"
        )

    # AUTOPILOT MODE: Enhanced handling for autopilot requests
    if request.autopilot:
        try:
            # Build enhanced autopilot prompt
            autopilot_prompt = _build_autopilot_prompt(
                last_customer_message=last_customer_message,
                conversation_context=vector_context or (format_conversation_context(messages) if messages else ""),
                recent_messages=request.recentMessages or [],
                system_prompt=request.systemPrompt,
                agent_name=request.agentName,
                customer_info=request.customerInfo,
                additional_context=request.context or ""
            )
            
            # Use specified model and temperature for autopilot
            llm = ChatOpenAI(
                model=request.model or "gpt-4o-mini",
                temperature=request.temperature or 0.7,
                **get_llm_config()
            )
            
            result = await llm.ainvoke(autopilot_prompt)
            response_text = result.content or "I'd be happy to help! Could you please provide more details?"
            
            # Calculate confidence score for autopilot
            confidence = _calculate_enhanced_autopilot_confidence(
                context=vector_context or "",
                last_message=last_customer_message,
                response=response_text,
                recent_messages=request.recentMessages or []
            )
            
            return ResponseSuggestionResponse(
                response_suggestion=response_text,
                response_suggestions=[response_text],
                total=1,
                conversationId=request.conversationId or "autopilot_response",
                autopilot_response=response_text if confidence >= 0.7 else None,
                confidence_score=confidence
            )
            
        except Exception as e:
            logger.error(f"Autopilot mode failed: {e}")
            # Fall through to regular processing

    # Try custom agent first if specified
    if request.aiAgentId:
        try:
            cached_agent_service = await get_cached_custom_agent_service()
            agent = await cached_agent_service.get_agent(request.aiAgentId, request.userId)
            if agent and agent.agentType == "response" and agent.isActive:
                context = {
                    "context": vector_context or (format_conversation_context(messages) if messages else ""),
                    "last_customer_message": last_customer_message,
                    "additional_context": request.context or "",
                    "mode": mode
                }
                
                result = await cached_agent_service.execute_agent(
                    request.aiAgentId, request.userId, context,
                    temperature=request.temperature,
                    model=request.model,
                    humanlike_behavior=request.humanlikeBehavior
                )
                
                # Handle autopilot mode
                if request.autopilot:
                    confidence = calculate_autopilot_confidence(
                        vector_context or "",
                        last_customer_message,
                        result
                    )
                    return ResponseSuggestionResponse(
                        response_suggestion=result,
                        response_suggestions=[result],
                        total=1,
                        conversationId=request.conversationId or "kb_query",
                        autopilot_response=result if confidence >= 0.7 else None,
                        confidence_score=confidence
                    )
                
                return ResponseSuggestionResponse(
                    response_suggestion=result,
                    response_suggestions=[result],
                    total=1,
                    conversationId=request.conversationId or "kb_query"
                )
        except Exception as e:
            logger.warning(f"Custom agent {request.aiAgentId} failed: {e}")

    # Intelligent AI suggestions using frontend agentInfo or intelligent defaults  
    if not request.aiAgentId:
        logger.info("No agent ID provided - generating intelligent AI response suggestions")
        
        # Use LLM to generate intelligent responses
        try:
            llm = ChatOpenAI(
                model=request.model or "gpt-4o-mini",
                temperature=request.temperature or 0.7,
                **get_llm_config()
            )
            
            # Build intelligent context-aware response
            context_text = vector_context or (format_conversation_context(messages) if messages else "")
            
            response_prompt = f"""You are a professional customer service representative generating a helpful response.

CONVERSATION CONTEXT:
{context_text or "No previous context available"}

CUSTOMER'S MESSAGE:
"{last_customer_message}"

TASK: Generate one helpful, professional response that a customer service representative would send to this customer.

REQUIREMENTS:
- Be conversational and ready to send
- Use the conversation context to make your response relevant and specific
- Be helpful, professional, and empathetic
- Address the customer's specific needs or questions
- Offer concrete next steps or assistance
- Sound natural and caring

Generate a single response:"""

            result = await llm.ainvoke(response_prompt)
            ai_response = result.content or "I'd be happy to help! Could you please provide more details?"
            
            logger.info("Intelligent AI generated response suggestion")
            return ResponseSuggestionResponse(
                response_suggestion=ai_response,
                response_suggestions=[ai_response],
                total=1,
                conversationId=request.conversationId or "kb_query"
            )
            
        except Exception as e:
            logger.warning(f"Intelligent AI response generation failed: {e}")
            # Fall through to enhanced agents or basic fallback
    else:
        # Frontend provided an agent ID but it wasn't found or failed above
        logger.warning(f"Frontend requested agent ID '{request.aiAgentId}' but it was not found or failed - using intelligent fallback")

    # Use enhanced agent if available (fallback) - only for conversation mode
    if ENHANCED_AGENTS_AVAILABLE and has_conversation:
        try:
            graph = create_response_suggestions_graph()
            result = await invoke_graph_safely(
                graph,
                {
                    "last_customer_message": last_customer_message,
                    "conversation_context": vector_context or format_conversation_context(messages),
                    "mode": mode
                }
            )
            
            response_suggestion = result.get("response_suggestion", "")
            if response_suggestion:
                # Handle autopilot mode
                if request.autopilot:
                    confidence = calculate_autopilot_confidence(
                        vector_context or format_conversation_context(messages),
                        last_customer_message,
                        response_suggestion
                    )
                    return ResponseSuggestionResponse(
                        response_suggestion=response_suggestion,
                        response_suggestions=[response_suggestion],
                        total=1,
                        conversationId=request.conversationId,
                        autopilot_response=response_suggestion if confidence >= 0.7 else None,
                        confidence_score=confidence
                    )
                
                return ResponseSuggestionResponse(
                    response_suggestion=response_suggestion,
                    response_suggestions=[response_suggestion],
                    total=1,
                    conversationId=request.conversationId
                )
        except Exception as e:
            logger.warning(f"Enhanced response agent failed: {e}")

    # Final fallback using basic LLM
    try:
        llm = ChatOpenAI(**get_llm_config())
        context_text = vector_context or (format_conversation_context(messages) if messages else "")
        
        if has_conversation:
            prompt = build_response_prompt(
                context=context_text,
                last_message=last_customer_message,
                additional_context=request.context or "",
                autopilot=request.autopilot
            )
        else:
            # KB-only mode prompt
            prompt = f"""Based on the following knowledge base information, provide a helpful response to the user's query.

Knowledge Base Context:
{context_text}

User Query: {last_customer_message}

Additional Context: {request.context or 'None'}

Provide a clear, helpful response based on the available information. If the information is not available in the knowledge base, politely indicate that."""

        result = await llm.ainvoke(prompt)
        response_text = result.content or "I'd be happy to help! Could you please provide more details?"
        
        # Handle autopilot mode
        if request.autopilot:
            confidence = calculate_autopilot_confidence(
                context_text,
                last_customer_message,
                response_text
            )
            return ResponseSuggestionResponse(
                response_suggestion=response_text,
                response_suggestions=[response_text],
                total=1,
                conversationId=request.conversationId or "kb_query",
                autopilot_response=response_text if confidence >= 0.7 else None,
                confidence_score=confidence
            )
        
        return ResponseSuggestionResponse(
            response_suggestion=response_text,
            response_suggestions=[response_text],
            total=1,
            conversationId=request.conversationId or "kb_query"
        )
        
    except Exception as e:
        logger.error(f"Final fallback failed: {e}")
        return ResponseSuggestionResponse(
            response_suggestion="I'd be happy to help! Could you please provide more details about what you need?",
            response_suggestions=["I'd be happy to help! Could you please provide more details about what you need?"],
            total=1,
            conversationId=request.conversationId or "kb_query"
        ) 