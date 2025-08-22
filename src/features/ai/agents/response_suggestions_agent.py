from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
import logging
from langgraph.graph import StateGraph, START, END
from ..config import get_llm_config

logger = logging.getLogger(__name__)

async def generate_response_suggestions(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a single response to a customer message.
    """
    try:
        llm_config = get_llm_config()
        if not llm_config.get("api_key"):
            logger.error("Missing OpenAI API key for response suggestions.")
            return {
                **state,
                "response_suggestions": ["Thank you for reaching out to us. I understand your concern and I'm here to help you resolve this."]
            }

        llm = ChatOpenAI(**llm_config)
        
        # Get conversation data
        recent_messages = state.get("recent_messages", [])
        vector_context = state.get("vector_context", "")
        last_customer_message = state.get("last_customer_message", "")
        
        # Build context
        context_parts = []
        
        if vector_context:
            context_parts.append(f"Conversation History:\n{vector_context}")
        
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
        
        if not full_context and not last_customer_message:
            return {
                **state,
                "response_suggestions": ["Hello! Thank you for contacting us. I'm here to help you with whatever you need today."]
            }

        # Create enhanced prompt for generating response suggestions
        prompt = f"""You are generating a ready-to-send response for a customer service representative. This response will be sent directly to the customer, so it must be clean, professional, and natural.

CONVERSATION CONTEXT:
{full_context}

CUSTOMER'S MESSAGE:
"{last_customer_message}"

CRITICAL FORMATTING REQUIREMENTS:
- NO markdown formatting (no **, ##, bullets, etc.)
- NO section headers or titles
- NO meta-commentary about what you're doing
- PLAIN TEXT ONLY that can be sent directly to the customer
- Sound natural and conversational
- Be a complete, helpful response
- Show understanding of the customer's situation
- Offer specific next steps or assistance

Write one complete response that sounds like it's coming from a helpful, knowledgeable customer service representative who genuinely cares about solving the customer's needs."""

        try:
            result = await llm.ainvoke(prompt)
            response_text = getattr(result, "content", "")
            
            if not response_text:
                raise ValueError("Empty response from LLM")
            
            # Parse the single response
            response_text = response_text.strip()
            
            if not response_text:
                raise ValueError("Empty response from LLM")
            
            # Clean up the response (remove any markdown formatting)
            cleaned_response = response_text.replace('**', '').replace('*', '').strip()
            cleaned_response = cleaned_response.replace('[', '').replace(']', '').strip()
            cleaned_response = cleaned_response.replace('](', ' - ').replace(')', '').strip()
            
            # Return single response in the expected format
            return {
                **state,
                "response_suggestions": [cleaned_response]  # Single response in array for compatibility
            }
            
        except Exception as llm_error:
            logger.error(f"LLM processing failed: {str(llm_error)}")
            # Return fallback suggestions
            return {
                **state,
                "response_suggestions": ["Thank you for contacting us about this. I understand your concern and I'm here to help you find a resolution."]
            }

    except Exception as error:
        logger.error(f"Error in generate_response_suggestions: {error}", exc_info=True)
        return {
            **state,
            "response_suggestions": ["Hello! Thank you for reaching out to us today. I'm here to help you with whatever you need."]
        }

def create_response_suggestions_graph() -> Any:
    """
    Create a simple graph for generating response suggestions.
    """
    try:
        graph = StateGraph(Dict[str, Any])
        graph.add_node("generate_suggestions", generate_response_suggestions)
        graph.add_edge(START, "generate_suggestions")
        graph.add_edge("generate_suggestions", END)
        return graph.compile()
    except Exception as e:
        logger.error(f"Failed to create response suggestions graph: {e}", exc_info=True)
        raise RuntimeError("Failed to create response suggestions graph") 