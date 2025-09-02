import os
import structlog
from prometheus_client import Counter, Histogram
import logging
import sys
from datetime import datetime
from typing import Dict, Any
import json
import asyncio

from ..ai.vector.vector_store import VectorStoreConfig, VectorStoreService

# Enhanced logging setup with debugging
def setup_enhanced_logging():
    """Setup enhanced logging with debug capabilities"""
    
    # Create custom formatter for debug logs
    class DebugFormatter(logging.Formatter):
        def format(self, record):
            # Add emojis and enhanced formatting for debug logs
            if hasattr(record, 'extra') and record.extra:
                record.msg = f"{record.msg} | Extra: {json.dumps(record.extra, default=str)}"
            
            # Handle Unicode characters safely
            try:
                formatted = super().format(record)
                return formatted
            except UnicodeEncodeError:
                # Fallback: remove emojis and special characters
                safe_msg = str(record.msg).encode('ascii', errors='ignore').decode('ascii')
                record.msg = safe_msg
                return super().format(record)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if os.getenv("DEBUG_LOGS", "false").lower() == "true" else logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler with enhanced formatting and Unicode support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Set UTF-8 encoding for console output
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
    
    # Enhanced format with better spacing and readability
    formatter = DebugFormatter(
        '%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Optional file handler for persistent logs
    if os.getenv("LOG_TO_FILE", "false").lower() == "true":
        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler(f"logs/vox_debug_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Call setup function
setup_enhanced_logging()

# Get logger for this module
logger = logging.getLogger(__name__)

# Add debugging utilities
def get_safe_debug_prefix(emoji: str, label: str) -> str:
    """Get a safe debug prefix that works on all systems"""
    import os
    import sys
    
    # Check if we're on Windows and console doesn't support UTF-8
    if os.name == 'nt':
        try:
            # Try to encode the emoji
            emoji.encode(sys.stdout.encoding or 'cp1252')
            return f"{emoji} [DEBUG] {label}"
        except (UnicodeEncodeError, LookupError):
            # Fall back to ASCII-safe version
            emoji_map = {
                "🔍": "[QUERY]",
                "🔥": "[PROMPT]", 
                "🤖": "[AI]",
                "🗄️": "[DB]",
                "📚": "[TRAIN]",
                "💡": "[SUGGEST]",
                "📝": "[RESPONSE]",
                "✅": "[SUCCESS]",
                "❌": "[ERROR]",
                "⚠️": "[WARN]",
                "📤": "[OUTPUT]",
                "🌐": "[REQUEST]"
            }
            safe_emoji = emoji_map.get(emoji, "[DEBUG]")
            return f"{safe_emoji} {label}"
    else:
        # Unix systems generally handle UTF-8 better
        return f"{emoji} [DEBUG] {label}"

def safe_emoji_log(message: str, level: str = "info", **kwargs):
    """Safely log messages with emoji fallback for Windows"""
    try:
        getattr(logger, level)(message, **kwargs)
    except UnicodeEncodeError:
        # Extract emoji and replace with safe version
        import re
        # Find emojis at start of message
        emoji_pattern = r'^([🔍🔥🤖🗄️📚💡📝✅❌⚠️📤🌐]+)'
        match = re.match(emoji_pattern, message)
        if match:
            emoji = match.group(1)
            rest_of_message = message[len(emoji):].strip()
            safe_prefix = get_safe_debug_prefix(emoji, "")
            safe_message = f"{safe_prefix}{rest_of_message}"
        else:
            # Remove all emojis and special characters
            safe_message = message.encode('ascii', errors='ignore').decode('ascii')
        
        getattr(logger, level)(safe_message, **kwargs)

def log_request_details(endpoint: str, request_data: Dict[str, Any], user_id: str = None):
    """Log comprehensive request details"""
    safe_emoji_log(f"🌐 [DEBUG] API REQUEST: {endpoint}", "info", extra={
        "endpoint": endpoint,
        "user_id": user_id,
        "request_size": len(str(request_data)),
        "timestamp": datetime.now().isoformat(),
        "request_data": request_data
    })

def log_response_details(endpoint: str, response_data: Dict[str, Any], processing_time: float = None):
    """Log comprehensive response details"""
    safe_emoji_log(f"📤 [DEBUG] API RESPONSE: {endpoint}", "info", extra={
        "endpoint": endpoint,
        "response_size": len(str(response_data)),
        "processing_time_ms": processing_time * 1000 if processing_time else None,
        "timestamp": datetime.now().isoformat(),
        "response_preview": str(response_data)[:200] + "..." if len(str(response_data)) > 200 else str(response_data)
    })

def log_llm_interaction(prompt: str, response: str, model: str = None, processing_time: float = None):
    """Log LLM interactions for debugging"""
    safe_emoji_log(f"🤖 [DEBUG] LLM INTERACTION", "info", extra={
        "model": model,
        "prompt_length": len(prompt),
        "response_length": len(response),
        "processing_time_ms": processing_time * 1000 if processing_time else None,
        "timestamp": datetime.now().isoformat()
    })
    safe_emoji_log(f"🤖 [DEBUG] PROMPT: {prompt[:500]}..." if len(prompt) > 500 else prompt, "debug")
    safe_emoji_log(f"🤖 [DEBUG] RESPONSE: {response[:500]}..." if len(response) > 500 else response, "debug")

def log_database_operation(operation: str, details: Dict[str, Any], success: bool = True):
    """Log database operations"""
    status = "SUCCESS" if success else "FAILED"
    safe_emoji_log(f"🗄️ [DEBUG] DB OPERATION {status}: {operation}", "info", extra={
        "operation": operation,
        "success": success,
        "details": details,
        "timestamp": datetime.now().isoformat()
    })
# Configure metrics for API monitoring - SIMPLE WORKING SOLUTION
try:
    API_REQUESTS = Counter('api_requests_total', 'Total API requests', ['endpoint'])
    API_ERRORS = Counter('api_errors_total', 'Total API errors', ['endpoint'])
    API_LATENCY = Histogram('api_latency_seconds', 'API latency in seconds', ['endpoint'])
except ValueError as e:
    if "Duplicated timeseries" in str(e):
        # Create dummy metrics that won't cause errors
        class DummyMetric:
            def __init__(self, name):
                self._name = name
            
            def labels(self, *args, **kwargs):
                return self
            
            def inc(self, *args, **kwargs):
                pass
            
            def observe(self, *args, **kwargs):
                pass
            
            def set(self, *args, **kwargs):
                pass
            
            def time(self):
                # Add time method that returns a context manager
                return self
            
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
            
            def __call__(self, *args, **kwargs):
                return self
        
        API_REQUESTS = DummyMetric('api_requests_total')
        API_ERRORS = DummyMetric('api_errors_total')
        API_LATENCY = DummyMetric('api_latency_seconds')
        logger.warning("Using dummy metrics due to duplication error")
    else:
        raise
    
# Default vector store configuration
def get_default_vector_config() -> VectorStoreConfig:
    """Get the default vector store configuration."""
    return VectorStoreConfig(
        voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
        model_name="voyage-3-large",
        embedding_dimension=1536,
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    ) 

# Service instance caching for performance
_service_cache = {}
_cache_lock = asyncio.Lock()

async def get_cached_vector_service(user_id: str = None) -> VectorStoreService:
    """Get cached vector service instance for performance."""
    cache_key = f"vector_service_{user_id or 'default'}"
    
    async with _cache_lock:
        if cache_key not in _service_cache:
            config = get_default_vector_config()
            _service_cache[cache_key] = VectorStoreService(config=config)
        
        return _service_cache[cache_key]

async def get_cached_custom_agent_service():
    """Get cached custom agent service instance for performance."""
    cache_key = "custom_agent_service"
    
    async with _cache_lock:
        if cache_key not in _service_cache:
            from ..ai.agents.custom_agent_service import CustomAgentService
            _service_cache[cache_key] = CustomAgentService()
        
        return _service_cache[cache_key]

async def get_cached_multi_kb_service():
    """Get cached multi-KB service instance for performance."""
    cache_key = "multi_kb_service"
    
    async with _cache_lock:
        if cache_key not in _service_cache:
            from ..ai.vector.multi_kb_service import MultiKnowledgeBaseService
            vector_config = get_default_vector_config()
            _service_cache[cache_key] = MultiKnowledgeBaseService(vector_config=vector_config)
        
        return _service_cache[cache_key]

def clear_service_cache():
    """Clear service cache (useful for testing or memory management)."""
    global _service_cache
    _service_cache.clear() 