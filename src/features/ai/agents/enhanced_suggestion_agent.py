from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
import logging
import json
from langgraph.graph import StateGraph, START, END
from ..config import get_llm_config

logger = logging.getLogger(__name__)

async def analyze_customer_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze customer context to understand their situation, needs, and emotional state.
    """
    try:
        llm_config = get_llm_config()
        if not llm_config.get("api_key"):
            logger.error("Missing OpenAI API key for context analysis.")
            return {
                **state,
                "context_analysis": {
                    "customer_intent": "unknown",
                    "emotional_state": "neutral",
                    "urgency_level": "medium",
                    "key_topics": [],
                    "pain_points": [],
                    "customer_type": "general"
                }
            }

        llm = ChatOpenAI(**llm_config)
        
        # Get conversation data
        recent_messages = state.get("recent_messages", [])
        vector_context = state.get("vector_context", "")
        last_customer_message = state.get("last_customer_message", "")
        
        # Build context for analysis
        context_parts = []
        if vector_context:
            context_parts.append(f"Historical Context:\n{vector_context}")
        
        if recent_messages:
            recent_context = []
            for msg in recent_messages[-5:]:
                role = msg.get("role", "unknown")
                body = msg.get("body", "")
                if body.strip():
                    recent_context.append(f"[{role}]: {body}")
            if recent_context:
                context_parts.append(f"Recent Messages:\n" + "\n".join(recent_context))
        
        full_context = "\n\n".join(context_parts) if context_parts else ""
        
        analysis_prompt = f"""You are an expert customer service analyst. Analyze the customer interaction below and provide a detailed assessment.

CONVERSATION CONTEXT:
{full_context}

LATEST CUSTOMER MESSAGE:
{last_customer_message}

Analyze and return a JSON object with the following structure:
{{
    "customer_intent": "seeking_information|requesting_help|making_complaint|expressing_satisfaction|asking_question|other",
    "emotional_state": "frustrated|worried|satisfied|confused|angry|neutral|happy",
    "urgency_level": "low|medium|high|critical",
    "key_topics": ["topic1", "topic2", "topic3"],
    "pain_points": ["pain1", "pain2"],
    "customer_type": "new_customer|returning_customer|premium_customer|technical_user|general_user",
    "conversation_stage": "initial_contact|information_gathering|problem_solving|resolution|follow_up",
    "requires_escalation": true/false,
    "sentiment_score": 0.0-1.0,
    "complexity_level": "simple|moderate|complex|very_complex"
}}

Focus on understanding what the customer really needs and how they're feeling about their situation."""

        try:
            result = await llm.ainvoke(analysis_prompt)
            analysis_text = getattr(result, "content", "{}")
            
            # Parse JSON response
            try:
                context_analysis = json.loads(analysis_text)
            except json.JSONDecodeError:
                # Fallback analysis if JSON parsing fails
                context_analysis = {
                    "customer_intent": "seeking_information",
                    "emotional_state": "neutral",
                    "urgency_level": "medium",
                    "key_topics": ["general inquiry"],
                    "pain_points": ["needs assistance"],
                    "customer_type": "general_user",
                    "conversation_stage": "information_gathering",
                    "requires_escalation": False,
                    "sentiment_score": 0.5,
                    "complexity_level": "moderate"
                }
            
            logger.info(f"Customer context analyzed - Intent: {context_analysis.get('customer_intent')}, Emotion: {context_analysis.get('emotional_state')}")
            
            return {
                **state,
                "context_analysis": context_analysis
            }
            
        except Exception as llm_error:
            logger.error(f"LLM analysis failed: {str(llm_error)}")
            return {
                **state,
                "context_analysis": {
                    "customer_intent": "seeking_information",
                    "emotional_state": "neutral",
                    "urgency_level": "medium",
                    "key_topics": ["general inquiry"],
                    "pain_points": ["needs assistance"],
                    "customer_type": "general_user"
                }
            }

    except Exception as error:
        logger.error(f"Error in analyze_customer_context: {error}", exc_info=True)
        return {
            **state,
            "context_analysis": {
                "customer_intent": "unknown",
                "emotional_state": "neutral",
                "urgency_level": "medium",
                "key_topics": [],
                "pain_points": [],
                "customer_type": "general"
            }
        }

async def generate_strategic_reasoning(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate strategic reasoning about how to best help the customer.
    """
    try:
        llm_config = get_llm_config()
        if not llm_config.get("api_key"):
            logger.error("Missing OpenAI API key for strategic reasoning.")
            return {
                **state,
                "strategic_reasoning": {
                    "approach": "standard_support",
                    "priorities": ["address_immediate_need"],
                    "communication_style": "professional",
                    "next_steps": ["gather_more_information"]
                }
            }

        llm = ChatOpenAI(**llm_config)
        
        context_analysis = state.get("context_analysis", {})
        recent_messages = state.get("recent_messages", [])
        last_customer_message = state.get("last_customer_message", "")
        
        reasoning_prompt = f"""You are a strategic customer service advisor. Based on the customer analysis below, determine the best approach to help this customer.

CUSTOMER ANALYSIS:
{json.dumps(context_analysis, indent=2)}

LATEST CUSTOMER MESSAGE:
{last_customer_message}

Consider the customer's emotional state, urgency level, and intent to determine the optimal strategy.

Return a JSON object with strategic recommendations:
{{
    "approach": "empathetic_support|technical_assistance|escalation_needed|information_provision|problem_solving",
    "priorities": ["priority1", "priority2", "priority3"],
    "communication_style": "empathetic|professional|technical|casual|urgent",
    "tone_adjustments": ["adjustment1", "adjustment2"],
    "next_steps": ["step1", "step2", "step3"],
    "escalation_recommended": true/false,
    "follow_up_needed": true/false,
    "estimated_resolution_time": "immediate|short|medium|long",
    "success_metrics": ["metric1", "metric2"]
}}

Focus on what will make this customer feel heard, understood, and properly assisted."""

        try:
            result = await llm.ainvoke(reasoning_prompt)
            reasoning_text = getattr(result, "content", "{}")
            
            try:
                strategic_reasoning = json.loads(reasoning_text)
            except json.JSONDecodeError:
                strategic_reasoning = {
                    "approach": "standard_support",
                    "priorities": ["address_immediate_need", "provide_clear_information"],
                    "communication_style": "professional",
                    "tone_adjustments": ["be_helpful", "stay_positive"],
                    "next_steps": ["gather_more_information", "provide_solution"],
                    "escalation_recommended": False,
                    "follow_up_needed": True,
                    "estimated_resolution_time": "medium",
                    "success_metrics": ["customer_satisfaction", "issue_resolution"]
                }
            
            logger.info(f"Strategic reasoning generated - Approach: {strategic_reasoning.get('approach')}")
            
            return {
                **state,
                "strategic_reasoning": strategic_reasoning
            }
            
        except Exception as llm_error:
            logger.error(f"Strategic reasoning LLM failed: {str(llm_error)}")
            return {
                **state,
                "strategic_reasoning": {
                    "approach": "standard_support",
                    "priorities": ["address_immediate_need"],
                    "communication_style": "professional",
                    "next_steps": ["provide_assistance"]
                }
            }

    except Exception as error:
        logger.error(f"Error in generate_strategic_reasoning: {error}", exc_info=True)
        return {
            **state,
            "strategic_reasoning": {
                "approach": "standard_support",
                "priorities": ["help_customer"],
                "communication_style": "professional",
                "next_steps": ["provide_support"]
            }
        }

async def generate_intelligent_suggestions(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate intelligent, contextually-aware suggestions based on analysis and reasoning.
    """
    try:
        llm_config = get_llm_config()
        if not llm_config.get("api_key"):
            logger.error("Missing OpenAI API key for suggestion generation.")
            return {
                **state,
                "suggestions": [
                    "Hi! I'm here to help you with whatever you need today.",
                    "Hello! I noticed you reached out. How can I assist you?",
                    "Hi there! I'm ready to help. What can I do for you?"
                ]
            }

        llm = ChatOpenAI(**llm_config)
        
        context_analysis = state.get("context_analysis", {})
        strategic_reasoning = state.get("strategic_reasoning", {})
        last_customer_message = state.get("last_customer_message", "")
        recent_messages = state.get("recent_messages", [])
        
        # Build comprehensive context
        context_parts = []
        if recent_messages:
            recent_context = []
            for msg in recent_messages[-3:]:  # Last 3 messages for focus
                role = msg.get("role", "unknown")
                body = msg.get("body", "")
                if body.strip():
                    recent_context.append(f"[{role}]: {body}")
            if recent_context:
                context_parts.append("Recent conversation:\n" + "\n".join(recent_context))
        
        full_context = "\n\n".join(context_parts) if context_parts else ""
        
        if not full_context and not last_customer_message:
            # Fallback responses when no context
            return {
                **state,
                "suggestions": [
                    "Hello! How can I help you today?",
                    "Hi there! I'm here to assist you with anything you need.",
                    "Welcome! What can I help you with?"
                ]
            }
        
        suggestion_prompt = f"""You are generating 3 ready-to-send response options for a customer service representative. These responses will be sent directly to customers, so they must be clean, professional, and natural.

CONVERSATION CONTEXT:
{chr(10).join(context_parts) if context_parts else "This is the start of the conversation"}

CUSTOMER'S MESSAGE:
"{last_customer_message}"

CRITICAL FORMATTING REQUIREMENTS:
- NO markdown formatting (no **, ##, bullets, etc.)
- NO section headers or titles  
- NO meta-commentary about what you're doing
- PLAIN TEXT ONLY that can be sent directly to the customer
- Sound natural and conversational
- Be complete, helpful responses
- Show understanding of the customer's situation
- Offer specific next steps or assistance

Generate exactly 3 different response options that the customer service representative can send directly to the customer:

1. [First response option - plain text, ready to send]
2. [Second response option - plain text, ready to send]  
3. [Third response option - plain text, ready to send]

Each response should be a complete message that sounds like it's coming from a helpful, knowledgeable customer service representative who genuinely cares about solving the customer's needs."""

        try:
            result = await llm.ainvoke(suggestion_prompt)
            response_text = getattr(result, "content", "")
            
            logger.info(f"LLM response received: {response_text[:200]}...")  # Log first 200 chars for debugging
            
            # Parse numbered suggestions with improved logic
            suggestions = []
            lines = response_text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                # Look for numbered lines (1., 2., 3.) and extract content
                if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                    suggestion = line[2:].strip()
                    # Skip if it's a placeholder or template text
                    if suggestion and not any(placeholder in suggestion.lower() for placeholder in [
                        'reminder message:', 'preparation message:', 'follow-up message:',
                        '[first suggestion]', '[second suggestion]', '[third suggestion]',
                        '**[', 'placeholder', 'your first', 'your second', 'your third',
                        'message:', 'response:', 'suggestion:'
                    ]):
                        suggestions.append(suggestion)
            
            # If we still don't have good suggestions, try alternative parsing
            if len(suggestions) < 3:
                logger.warning(f"Only got {len(suggestions)} suggestions from numbered parsing, trying alternative methods")
                
                # Try to extract sentences that look like suggestions
                sentences = [s.strip() for s in response_text.replace('\n', '. ').split('.') if s.strip()]
                for sentence in sentences:
                    if (len(sentence) > 20 and 
                        not any(placeholder in sentence.lower() for placeholder in [
                            '[first', '[second', '[third', '**[', 'placeholder'
                        ]) and
                        len(suggestions) < 3):
                        suggestions.append(sentence.strip())
            
            # Fallback if parsing still fails
            if len(suggestions) < 3:
                logger.warning("Enhanced suggestion parsing failed, using contextual fallbacks")
                emotional_state = context_analysis.get('emotional_state', 'neutral')
                customer_intent = context_analysis.get('customer_intent', 'seeking_information')
                
                if emotional_state == 'frustrated':
                    suggestions = [
                        "I understand this has been frustrating for you. Let me help resolve this right away.",
                        "I can see this situation is causing you stress. I'm here to make this right for you.",
                        "I want to help fix this issue for you as quickly as possible."
                    ]
                elif emotional_state == 'worried':
                    suggestions = [
                        "I can help put your mind at ease about this situation.",
                        "Let me provide you with the information you need to feel more confident about this.",
                        "I'm here to help address your concerns and provide clarity."
                    ]
                elif emotional_state == 'confused':
                    suggestions = [
                        "Let me help clarify this situation for you.",
                        "I can break this down and make it easier to understand.",
                        "I'm here to help explain this step by step."
                    ]
                elif customer_intent == 'requesting_help':
                    suggestions = [
                        "I'm here to help you with this. Let me assist you right away.",
                        "I can definitely help you with that. Let me get started on this for you.",
                        "I'm ready to help you resolve this. Let me take care of this for you."
                    ]
                elif customer_intent == 'making_complaint':
                    suggestions = [
                        "I understand your concern and I want to make this right for you.",
                        "I'm sorry this happened. Let me work on resolving this for you immediately.",
                        "I take this seriously and I'm committed to fixing this for you."
                    ]
                else:
                    suggestions = [
                        "I'm here to help you with whatever you need.",
                        "How can I best assist you today?",
                        "I'm ready to help. What would be most useful for you?"
                    ]
            
            # Ensure we have exactly 3 suggestions and clean them up
            suggestions = suggestions[:3]
            while len(suggestions) < 3:
                suggestions.append("How else can I help you today?")
            
            # Clean up suggestions (remove any remaining formatting)
            cleaned_suggestions = []
            for suggestion in suggestions:
                cleaned = suggestion.strip()
                # Remove any markdown formatting or asterisks
                cleaned = cleaned.replace('**', '').replace('*', '').strip()
                # Remove any remaining brackets
                cleaned = cleaned.replace('[', '').replace(']', '').strip()
                cleaned_suggestions.append(cleaned)
            
            logger.info(f"Generated {len(cleaned_suggestions)} intelligent suggestions successfully")
            
            return {
                **state,
                "suggestions": cleaned_suggestions
            }
            
        except Exception as llm_error:
            logger.error(f"Suggestion generation LLM failed: {str(llm_error)}")
            return {
                **state,
                "suggestions": [
                    "I'm here to help you with this situation.",
                    "Let me assist you with what you need.",
                    "How can I best help you today?"
                ]
            }

    except Exception as error:
        logger.error(f"Error in generate_intelligent_suggestions: {error}", exc_info=True)
        return {
            **state,
            "suggestions": [
                "Hello! How can I help you today?",
                "I'm here to assist you with whatever you need.",
                "What can I help you with?"
            ]
        }

def create_enhanced_suggestion_graph() -> Any:
    """
    Create an enhanced multi-step suggestion generation graph with reasoning capabilities.
    """
    try:
        graph = StateGraph(Dict[str, Any])
        
        # Add nodes for multi-step reasoning
        graph.add_node("analyze_context", analyze_customer_context)
        graph.add_node("strategic_reasoning", generate_strategic_reasoning)
        graph.add_node("generate_suggestions", generate_intelligent_suggestions)
        
        # Create the workflow
        graph.add_edge(START, "analyze_context")
        graph.add_edge("analyze_context", "strategic_reasoning")
        graph.add_edge("strategic_reasoning", "generate_suggestions")
        graph.add_edge("generate_suggestions", END)
        
        return graph.compile()
    except Exception as e:
        logger.error(f"Failed to create enhanced suggestion graph: {e}", exc_info=True)
        raise RuntimeError("Failed to create enhanced suggestion graph") 