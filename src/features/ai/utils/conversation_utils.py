import re
from typing import List

FALLBACK_SUGGESTIONS = [
    "I'm here to help - what's the most important thing I can do for you right now?",
    "What would make the biggest difference for you in this situation?",
    "I want to make sure I'm giving you exactly what you need - what matters most to you?"
]

FALLBACK_RESPONSES = [
    "I can see this is important to you, and I want to help you get it sorted out. Let me work on this with you."
]

def format_conversation_context(messages: List[dict]) -> str:
    """Format messages into a readable conversation context."""
    lines = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        direction = msg.get("direction", "")
        body = msg.get("body", "").strip()
        if not body:
            continue
        display_role = f"{role} ({direction})" if direction else role
        lines.append(f"{i+1}. [{display_role}]: {body}")
    return "\n".join(lines)

def build_suggestion_prompt(context: str, additional_context: str, limit: int) -> str:
    """Build prompt for generating clean, ready-to-send customer service responses."""

    return f"""You are generating {limit} ready-to-send responses for a customer service representative. These responses will be sent directly to customers, so they must be professional, helpful, and natural.

CONVERSATION CONTEXT:
{context}

ADDITIONAL CONTEXT: {additional_context or "None"}

Generate {limit} different response options that the customer service representative can send directly to the customer. Each response should:

CRITICAL FORMATTING REQUIREMENTS:
- NO markdown formatting (no **, ##, bullets, etc.)
- NO section headers or titles
- NO meta-commentary about what you're doing
- PLAIN TEXT ONLY that can be sent directly to the customer
- Sound natural and conversational
- Be complete, helpful responses
- Show understanding of the customer's situation
- Offer specific next steps or assistance

Return ONLY the numbered responses, nothing else:

1. [First response option - plain text, ready to send]
2. [Second response option - plain text, ready to send]
3. [Third response option - plain text, ready to send]

Each response should be a complete message that sounds like it's coming from a helpful, knowledgeable customer service representative who genuinely cares about solving the customer's needs."""

def build_response_prompt(context: str, last_message: str, additional_context: str, autopilot: bool = False) -> str:
    """Build human, conversational prompt for generating caring responses."""
    autopilot_note = """
IMPORTANT: This response will be sent automatically, so it needs to be perfect - warm, helpful, and exactly what this person needs to hear right now.
""" if autopilot else ""

    return f"""You're a real person who works in customer service because you genuinely love helping people. You're having a conversation with someone who needs your help.

CONVERSATION SO FAR:
{context}

WHAT THEY JUST TOLD YOU:
"{last_message}"

EXTRA CONTEXT: {additional_context or "Nothing additional"}

{autopilot_note}

Now respond like a real human being would - someone who:
- Actually cares about this person and their problem
- Listens to what they're really saying (and feeling)
- Speaks naturally, like you're talking to someone you care about
- Shows genuine empathy and understanding
- Offers real, practical help
- Makes them feel heard, valued, and taken care of

Think about how you'd want someone to respond to you if you were in their exact situation. What would make you feel better? What would show you that they really care and want to help?

Write a response that feels human, caring, and genuinely helpful - not scripted or robotic:"""

def extract_numbered_items(text: str, limit: int = 3) -> List[str]:
    """Extract numbered items from LLM response text, including multi-line content."""
    if not text:
        return []
    
    # Split by numbered patterns (1., 2., 3., etc.)
    # This regex splits on lines that start with a number followed by a period
    parts = re.split(r'\n(?=\d+\.)', text.strip())
    
    suggestions = []
    for i, part in enumerate(parts):
        if i >= limit:
            break
            
        # Clean up the part
        part = part.strip()
        if not part:
            continue
            
        # Remove the number prefix (1., 2., etc.) from the beginning
        cleaned = re.sub(r'^\d+\.\s*', '', part)
        
        # Clean up extra whitespace and newlines
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)  # Normalize paragraph breaks
        cleaned = re.sub(r'\n\s+', ' ', cleaned)  # Join broken lines
        cleaned = cleaned.strip()
        
        if cleaned:
            suggestions.append(cleaned)
    
    return suggestions

def calculate_autopilot_confidence(context: str, last_message: str, response: str) -> float:
    """Calculate enhanced confidence score for autopilot response."""
    confidence = 0.6  # Base confidence
    
    # Analyze response quality factors
    response_lower = response.lower()
    
    # Boost confidence for empathetic language
    empathy_indicators = [
        "understand", "appreciate", "sorry", "apologize", "feel", 
        "concern", "important", "priority", "care", "help"
    ]
    empathy_score = sum(1 for indicator in empathy_indicators if indicator in response_lower)
    confidence += min(0.15, empathy_score * 0.03)
    
    # Boost confidence for solution-oriented language
    solution_indicators = [
        "resolve", "solution", "fix", "address", "handle", 
        "take care", "ensure", "make sure", "will", "can"
    ]
    solution_score = sum(1 for indicator in solution_indicators if indicator in response_lower)
    confidence += min(0.15, solution_score * 0.03)
    
    # Boost confidence for personalized responses
    if last_message:
        customer_words = set(last_message.lower().split())
        response_words = set(response_lower.split())
        overlap = len(customer_words.intersection(response_words))
        if overlap > 2:  # More than just common words
            confidence += min(0.1, overlap * 0.02)
    
    # Boost confidence for appropriate length and structure
    if 50 <= len(response) <= 300:  # Optimal length range
        confidence += 0.05
    
    # Check for professional structure (sentences, proper grammar indicators)
    sentence_count = len([s for s in response.split('.') if s.strip()])
    if 2 <= sentence_count <= 5:  # Good structure
        confidence += 0.05
    
    # Reduce confidence for generic or template-like responses
    generic_phrases = [
        "thank you for contacting us", "we appreciate your business",
        "have a great day", "please let us know", "feel free to contact"
    ]
    generic_count = sum(1 for phrase in generic_phrases if phrase in response_lower)
    confidence -= generic_count * 0.05
    
    # Reduce confidence if response seems too short or too long
    if len(response) < 30:
        confidence -= 0.1
    elif len(response) > 400:
        confidence -= 0.05
    
    return min(1.0, max(0.0, confidence)) 