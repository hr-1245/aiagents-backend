"""
Prompt templates and output parsers for conversation management.
"""

from typing import List, Dict, Any, Literal, Union
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

def prepare_messages(messages: Union[str, List[Dict[str, Any]], None]) -> str:
    """Format conversation messages for template input."""
    if not messages:
        return "No conversation history available."
    if isinstance(messages, str):
        return messages
    
    formatted = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or msg.get("body", "")
        if content:
            formatted.append(f"{role}: {content}")
    
    return "\n".join(formatted) if formatted else "No valid messages available."

def prepare_context(context: Union[str, List[str], Dict[str, Any], None]) -> str:
    """Format context data for template input."""
    if context is None:
        return "No context available."
    if isinstance(context, str):
        return context
    if isinstance(context, list):
        return "\n".join(str(item) for item in context if item)
    if isinstance(context, dict):
        return "\n".join(f"{k}: {v}" for k, v in context.items() if v)
    return str(context)

# Output schema for query classification
class QueryType(BaseModel):
    type: Literal["technical", "general", "support", "summary", "autopilot", "invalid"] = Field(
        description="The classified type of the query"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )

query_classifier_parser = JsonOutputParser(pydantic_object=QueryType)

query_classifier_prompt = PromptTemplate(
    output_parser=query_classifier_parser,
    input_variables=["query"],
    template="""Analyze the query to determine its type and return a JSON response with 'type' and 'confidence' fields.
Valid types are: 'technical', 'general', 'support', 'summary', 'autopilot', or 'invalid'.

- 'technical' for technical topics (e.g., coding, algorithms)
- 'general' for non-technical queries
- 'support' for customer service queries
- 'summary' for summarization requests
- 'autopilot' for conversational bot requests
- 'invalid' for unclear queries

Query: {query}"""
)

# Output schema for context analysis
class ConversationAnalysis(BaseModel):
    topics: List[str] = Field(description="Key topics discussed")
    decisions: List[str] = Field(description="Important decisions or agreements made")
    questions: List[str] = Field(description="Explicit questions needing answers")
    implicit_concerns: List[str] = Field(description="Implicit questions or concerns")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Overall sentiment")
    relevant_context: str = Field(description="Key context relevant to the query")

context_analysis_parser = JsonOutputParser(pydantic_object=ConversationAnalysis)

context_analyzer_prompt = PromptTemplate(
    input_variables=["messages", "query"],
    partial_variables={"format": prepare_messages},
    output_parser=context_analysis_parser,
    template="""Analyze the conversation history and query to identify key elements.
Return a JSON object with the following structure:
{
    "topics": ["topic1", "topic2"],
    "decisions": ["decision1", "decision2"],
    "questions": ["question1", "question2"],
    "implicit_concerns": ["concern1", "concern2"],
    "sentiment": "positive/negative/neutral",
    "relevant_context": "summary of relevant context"
}

Conversation:
{format(messages)}

Query: {query}"""
)

# Output schema for reasoning plan
class ReasoningPlan(BaseModel):
    key_points: List[str] = Field(description="Key points to address")
    context_relevance: str = Field(description="How context informs the response")
    clarifications: List[str] = Field(description="Potential clarifications needed")
    approach: Dict[str, str] = Field(description="Response approach details")

reasoning_parser = JsonOutputParser(pydantic_object=ReasoningPlan)

reasoning_prompt = PromptTemplate(
    input_variables=["analysis", "query"],
    output_parser=reasoning_parser,
    template="""Based on the analysis, create a reasoning plan. Return as JSON with:
{
    "key_points": ["point1", "point2"],
    "context_relevance": "how context informs response",
    "clarifications": ["clarification1", "clarification2"],
    "approach": {
        "tone": "formal/casual/technical",
        "style": "detailed/concise/conversational"
    }
}

Analysis: {analysis}
Query: {query}"""
)

rag_query_prompt = PromptTemplate(
    input_variables=["custom_prompt", "context", "analysis", "reasoning", "query"],
    partial_variables={"format": prepare_context},
    template="""
{custom_prompt}

Context:
{format(context)}

Analysis:
{analysis}

Reasoning:
{reasoning}

Query: {query}"""
)

support_prompt = PromptTemplate(
    input_variables=["query", "context"],
    partial_variables={"format": prepare_context},
    template="""You are a system user/customer service representative responding to a customer. Please provide a polite and concise response from your perspective as the system user helping the customer.

Context (if any):
{format(context)}

Customer Query: {query}

Remember: You are the system user responding TO the customer, not the customer asking questions."""
)

summary_prompt = PromptTemplate(
    input_variables=["messages", "context"],
    partial_variables={
        "format_msg": prepare_messages,
        "format_ctx": prepare_context
    },
    template="""Provide a concise summary highlighting key points and outcomes.

Conversation:
{format_msg(messages)}

Additional Context:
{format_ctx(context)}"""
)

autopilot_prompt = PromptTemplate(
    input_variables=["messages", "query", "context", "analysis"],
    partial_variables={
        "format_msg": prepare_messages,
        "format_ctx": prepare_context
    },
    template="""You are a system user/customer service representative in autopilot mode, responding to customers. Maintain natural dialogue flow and adapt to customer sentiment while always responding from your perspective as the system user.

Conversation History:
{format_msg(messages)}

Context:
{format_ctx(context)}

Analysis:
{analysis}

Customer Query: {query}

Remember: You are the system user responding TO the customer, not the customer asking questions. Provide helpful, professional responses that move the conversation forward constructively."""
)

class SuggestionItem(BaseModel):
    question: str = Field(description="Follow-up question text")
    type: Literal["clarification", "exploration", "confirmation"] = Field(description="Question type")
    relevance: float = Field(description="Relevance score", ge=0.0, le=1.0)

suggestion_parser = JsonOutputParser(pydantic_object=List[SuggestionItem])

suggestion_prompt = PromptTemplate(
    input_variables=["messages", "response", "query_type"],
    partial_variables={"format": prepare_messages},
    output_parser=suggestion_parser,
    template="""You are a system user/customer service representative. Generate 3 follow-up questions that YOU would ask the customer to better assist them. Return as a JSON array of objects with:
{
    "question": "question text",
    "type": "clarification/exploration/confirmation",
    "relevance": 0.0-1.0
}

Conversation:
{format(messages)}

Your Last Response: {response}
Query Type: {query_type}

Generate questions that YOU (as the system user) would ask TO the customer to help them better."""
)

# Export components
__all__ = [
    # Prompts
    "query_classifier_prompt",
    "context_analyzer_prompt",
    "reasoning_prompt",
    "rag_query_prompt", 
    "support_prompt",
    "summary_prompt",
    "autopilot_prompt",
    "suggestion_prompt",
    
    # Parsers
    "query_classifier_parser",
    "context_analysis_parser",
    "reasoning_parser",
    "suggestion_parser",
    
    # Utilities
    "prepare_messages",
    "prepare_context",
    
    # Types
    "QueryType",
    "ConversationAnalysis",
    "ReasoningPlan",
    "SuggestionItem"
]