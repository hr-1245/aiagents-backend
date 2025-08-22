"""
Configuration settings for AI features.
"""

import os
from typing import Dict, Any

# OpenAI API configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

# Parse temperature with error handling
try:
    OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0"))
    if not 0 <= OPENAI_TEMPERATURE <= 2:
        print(f"Warning: Invalid temperature {OPENAI_TEMPERATURE}, using default 0")
        OPENAI_TEMPERATURE = 0
except ValueError:
    print(f"Warning: Invalid temperature value in environment, using default 0")
    OPENAI_TEMPERATURE = 0

# Voyage AI configuration
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")
VOYAGE_MODEL = os.environ.get("VOYAGE_MODEL", "voyage-3-large")

# ChromaDB configuration
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_PREFIX = os.environ.get("CHROMA_COLLECTION_PREFIX", "user_")

# LangGraph configuration
LANGGRAPH_DEBUG = os.environ.get("LANGGRAPH_DEBUG", "false").lower() == "true"


# Parse numeric settings with error handling
def get_int_env(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        print(f"Warning: Invalid value for {key}, using default {default}")
        return default


# Default RAG parameters
DEFAULT_RETRIEVAL_K = get_int_env("DEFAULT_RETRIEVAL_K", 5)
MAX_RETRIEVAL_LIMIT = get_int_env("MAX_RETRIEVAL_LIMIT", 100)

# System settings
DEFAULT_CHUNK_SIZE = get_int_env("DEFAULT_CHUNK_SIZE", 1000)
DEFAULT_CHUNK_OVERLAP = get_int_env("DEFAULT_CHUNK_OVERLAP", 200)


def get_llm_config() -> Dict[str, Any]:
    """
    Get the configuration for language models.
    Returns a dictionary with settings for the LLM.
    
    Returns:
        Dict with validated LLM configuration settings
    """
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not set")

    config = {
        "model": OPENAI_DEFAULT_MODEL or "gpt-4o",
        "temperature": OPENAI_TEMPERATURE or 0.0,
        "api_key": OPENAI_API_KEY,
    }

    # Validate model name
    # valid_models = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4o"]
    # if config["model"] not in valid_models:
    #     print(f"Warning: Invalid model {config['model']}, falling back to gpt-4")
    #     config["model"] = "gpt-4"

    return config


def get_dynamic_llm_config(temperature: float = None, model: str = None) -> Dict[str, Any]:
    """
    Get the configuration for language models with dynamic parameters.
    
    Args:
        temperature: AI response creativity/randomness (0.0-2.0)
        model: AI model to use (e.g., 'gpt-4o', 'gpt-4-turbo')
    
    Returns:
        Dict with validated LLM configuration settings
    """
    # Start with default config
    config = get_llm_config()
    
    # Override with provided parameters
    if temperature is not None:
        # Validate temperature: must be 0, 1, or between 0.1-0.9
        if temperature == 0.0 or temperature == 1.0 or (0.1 <= temperature <= 0.9):
            config["temperature"] = temperature
        else:
            print(f"Warning: Invalid temperature {temperature} (must be 0, 1, or 0.1-0.9), using default {config['temperature']}")
    
    if model is not None and model.strip():
        # Basic model validation - accept most model names
        model = model.strip()
        if model:
            config["model"] = model
        else:
            print(f"Warning: Empty model name provided, using default {config['model']}")
    
    return config


def get_humanlike_behavior_instructions() -> str:
    """
    Get instructions to make AI responses more human-like.
    
    Returns:
        String with human-like behavior instructions
    """
    return """
HUMAN-LIKE BEHAVIOR INSTRUCTIONS:
- Use natural, conversational language rather than overly formal or robotic responses
- Occasionally use filler words like "well", "you know", "actually" when appropriate
- Show empathy and understanding with phrases like "I understand how that can be frustrating"
- Use contractions naturally (don't → don't, it's → it's, I'll → I'll)
- Vary sentence structure and length to sound more natural
- Include gentle acknowledgments like "That's a great question" or "I can definitely help with that"
- Use transitional phrases like "Let me help you with that" or "Here's what I can tell you"
- Show personality while remaining professional and helpful
- Ask follow-up questions when clarification might be helpful
- Use inclusive language and avoid being overly technical unless necessary
"""
