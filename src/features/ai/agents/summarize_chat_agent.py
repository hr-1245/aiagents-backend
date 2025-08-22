from typing import Dict, Any
from langchain_openai import ChatOpenAI
import logging

from langgraph.graph import StateGraph, START, END
from ..config import get_llm_config

logger = logging.getLogger(__name__)


async def summarize_chat(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarizes the entire chat conversation using retrieved context.
    """
    try:
        llm_config = get_llm_config()
        if not llm_config.get("api_key"):
            logger.error("Missing OpenAI API key.")
            return {
                "summary": "Unable to generate summary due to missing API key."
            }

        llm = ChatOpenAI(**llm_config)

        chat_context = state.get("retrieved_context", "").strip()
        if not chat_context:
            return {
                "summary": "No conversation context available to summarize."
            }

        prompt = (
            "Summarize the following conversation clearly and concisely. "
            "Focus on the key points, user intent, and main takeaways.\n\n"
            f"Conversation:\n{chat_context}\n\n"
            "Summary:"
        )

        try:
            result = await llm.ainvoke(prompt)
            summary_text = getattr(result, "content", str(result))
        except Exception as llm_error:
            logger.error(f"LLM call failed during summarization: {llm_error}")
            summary_text = "An error occurred while generating the summary."

        return {
            **state,
            "summary": summary_text,
        }

    except Exception as error:
        logger.error(f"Error in summarize_chat: {error}", exc_info=True)
        return {
            "summary": "Unexpected error during summarization."
        }


def create_summary_graph() -> Any:
    """
    Creates a LangGraph with a single summarization node.
    """
    try:
        graph = StateGraph(Dict[str, Any])
        graph.add_node("summarize", summarize_chat)
        graph.add_edge(START, "summarize")
        graph.add_edge("summarize", END)
        return graph.compile()
    except Exception as e:
        logger.error(f"Summary graph creation failed: {e}", exc_info=True)
        raise RuntimeError("Failed to create summary graph")
