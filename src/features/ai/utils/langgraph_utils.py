"""
Utility functions for working with LangGraph.
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

async def invoke_graph_safely(graph: Any, state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Safely invoke a LangGraph workflow, handling all possible errors.
    
    Args:
        graph: The LangGraph compiled graph
        state: The state dictionary to pass to the graph
        **kwargs: Additional arguments (ignored for compatibility)
        
    Returns:
        Dict containing the response and other state fields
    """
    try:
        # Direct invocation with no extra parameters for maximum compatibility
        return await graph.ainvoke(state)
    except AttributeError as e:
        logger.warning(f"Async invoke not available: {e}")
        try:
            # Try synchronous invoke as fallback
            return graph.invoke(state)
        except Exception as e2:
            logger.error(f"All graph invocation methods failed: {e} / {e2}")
            return {
                "response": "I encountered a technical issue and couldn't process your request.",
                "suggestions": [],
                "summary": "Technical error occurred."
            }
    except Exception as e:
        logger.error(f"Error invoking graph: {e}")
        return {
            "response": "I encountered a technical issue and couldn't process your request.",
            "suggestions": [],
            "summary": "Technical error occurred."
        }
