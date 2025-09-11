import uvicorn
from fastapi import FastAPI, Depends, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import logging
from dotenv import load_dotenv
import os
import sys
import signal
from typing import Dict, Any, Optional

# Set protobuf environment variable early before any imports
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Enable UTF-8 console output on Windows
if os.name == "nt":
    try:
        # Try to set console to UTF-8 mode
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)  # UTF-8
        os.environ["PYTHONIOENCODING"] = "utf-8"
    except:
        # If that fails, we'll fall back to ASCII-safe logging
        pass

# Import enhanced logging configuration
try:
    from .features.server.config import setup_enhanced_logging

    setup_enhanced_logging()
    logger = logging.getLogger(__name__)
    logger.info("Enhanced debug logging system initialized")
except ImportError:
    # Fallback to basic logging if enhanced logging fails
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.warning("Using basic logging - enhanced debug features not available")

# Load environment variables
load_dotenv()

# Check for required environment variables
required_env_vars = ["OPENAI_API_KEY", "VOYAGE_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    error_message = f"Missing required environment variables: {', '.join(missing_vars)}"
    logger.error(error_message)
    logger.error(
        "Please create a .env file with the required variables or set them in your environment. Server cannot start."
    )
    sys.exit(1)  # Exit if critical environment variables are missing

# Initialize FastAPI app
app = FastAPI(
    title="VOX Backend API",
    description="Secure conversation management API with AI agents",
    version="1.0.0",
)

# Import and setup security
try:
    # Try absolute import first (when running from src directory)
    try:
        from features.server.security import (
            create_cors_config,
            authenticate,
            check_rate_limit,
        )
    except ModuleNotFoundError:
        # Try relative import (when running from project root via main.py)
        from src.features.server.security import (
            create_cors_config,
            authenticate,
            check_rate_limit,
        )

    security_loaded = True
    logger.info("Security module loaded successfully")

    # Configure CORS
    cors_config = create_cors_config()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config["allow_origins"],
        allow_credentials=cors_config["allow_credentials"],
        allow_methods=cors_config["allow_methods"],
        allow_headers=cors_config["allow_headers"],
        expose_headers=cors_config["expose_headers"],
    )
    logger.info("CORS middleware configured")

except Exception as e:
    logger.error(f"Error loading security module: {str(e)}")
    security_loaded = False
    logger.warning("Security module not loaded. API will run without authentication.")

    # Basic CORS setup if security module fails
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Not secure - for development only
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Safe logging utility for Unicode characters
def safe_log_text(text: str, max_length: int = 500) -> str:
    """Make text safe for logging by removing/replacing problematic Unicode characters."""
    if not text:
        return text

    try:
        # Truncate first
        if len(text) > max_length:
            text = text[:max_length] + "..."

        # Remove or replace problematic Unicode characters
        safe_text = ""
        for char in text:
            # Skip emojis and other problematic Unicode characters
            if ord(char) > 127:
                # Replace with safe equivalent or skip
                if ord(char) >= 0x1F600 and ord(char) <= 0x1F64F:  # Emoticons
                    safe_text += "[emoji]"
                elif ord(char) >= 0x1F300 and ord(char) <= 0x1F5FF:  # Misc symbols
                    safe_text += "[symbol]"
                elif ord(char) >= 0x1F680 and ord(char) <= 0x1F6FF:  # Transport symbols
                    safe_text += "[symbol]"
                elif ord(char) >= 0x2600 and ord(char) <= 0x26FF:  # Misc symbols
                    safe_text += "[symbol]"
                elif char in '""' "":  # Smart quotes
                    safe_text += '"'
                elif char == "—":  # Em dash
                    safe_text += "--"
                elif char == "…":  # Ellipsis
                    safe_text += "..."
                else:
                    safe_text += char  # Keep other Unicode chars like accented letters
            else:
                safe_text += char

        return safe_text
    except Exception:
        # Fallback: encode to ASCII and ignore errors
        return text.encode("ascii", "ignore").decode("ascii")


# Simple HTTP request logging (only if debug logging is enabled)
if os.environ.get("DEBUG_LOGS", "").lower() == "true":

    @app.middleware("http")
    async def log_requests(request, call_next):
        import time
        import json

        start_time = time.time()

        # Log HTTP request with clear formatting
        logger.info("=" * 60)
        logger.info(f"HTTP REQUEST: {request.method} {request.url.path}")

        # Show request body for POST/PUT (keep it simple)
        if (
            request.method in ["POST", "PUT", "PATCH"]
            and os.environ.get("LOG_REQUEST_BODY", "true").lower() == "true"
        ):
            try:
                body = await request.body()
                if body:
                    body_str = body.decode()
                    safe_body = safe_log_text(
                        body_str, 500
                    )  # First 500 chars only, Unicode-safe
                    logger.debug(f"  REQUEST BODY: {safe_body}")
            except Exception as e:
                logger.debug(f"  REQUEST BODY: [Could not decode body: {str(e)}]")

        # Process request
        response = await call_next(request)

        # Log response with timing
        process_time = time.time() - start_time
        logger.info(f"HTTP RESPONSE: {response.status_code} | {process_time:.2f}s")
        logger.info("=" * 60)

        return response


try:
    # Try absolute import first (when running from src directory)
    try:
        from features.server.api import router
    except ModuleNotFoundError:
        # Try relative import (when running from project root via main.py)
        from src.features.server.api import router

    api_loaded = True
    # Include API routes
    app.include_router(router, prefix="/ai/conversation")
    logger.info("API router loaded successfully")
except Exception as e:
    logger.error(f"Error loading API router: {str(e)}")
    import traceback

    logger.error(f"Full traceback: {traceback.format_exc()}")
    api_loaded = False
    logger.warning("API router not loaded. Only health endpoint will be available.")


def check_db_connection() -> bool:
    """Check if ChromaDB is accessible"""
    try:
        # Try absolute import first (when running from src directory)
        try:
            from features.ai.vector.vector_store import VectorStoreService
        except ModuleNotFoundError:
            # Try relative import (when running from project root via main.py)
            from src.features.ai.vector.vector_store import VectorStoreService

        vector_service = VectorStoreService()
        return vector_service.client is not None
    except Exception:
        return False


def signal_handler(signum: int, frame: Optional[Any]) -> None:
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}")
    logger.info("Shutting down server...")
    sys.exit(0)


def run_server() -> None:
    """Run the FastAPI server with proper configuration and signal handling"""
    try:
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info("Starting FastAPI server")

        # Ensure persist directory exists
        persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
        os.makedirs(persist_dir, exist_ok=True)
        logger.info(f"CHROMA_PERSIST_DIR is set to: {persist_dir}")

        # Check for potential protobuf issues
        if not os.environ.get("PROTOBUF_FIXED", ""):
            logger.warning(
                "If you encounter protobuf errors, try one of these solutions:"
            )
            logger.warning("1. pip install protobuf==3.20.3")
            logger.warning(
                "2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python in your environment"
            )

        # Get server configuration from environment
        host = os.environ.get("HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", "8000"))
        log_level = os.environ.get("LOG_LEVEL", "info").lower()

        # Configure uvicorn with performance optimizations
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level=log_level,
            reload=os.environ.get("DEBUG", "").lower() == "true",
            access_log=True,
            timeout_keep_alive=int(
                os.environ.get("KEEP_ALIVE", "30")
            ),  # Increased for better connection reuse
            workers=int(
                os.environ.get("WORKERS", "1")
            ),  # Can be increased for production
            limit_concurrency=int(
                os.environ.get("LIMIT_CONCURRENCY", "1000")
            ),  # Handle more concurrent requests
            limit_max_requests=int(
                os.environ.get("LIMIT_MAX_REQUESTS", "10000")
            ),  # Higher request limit
            timeout_graceful_shutdown=int(
                os.environ.get("GRACEFUL_SHUTDOWN", "30")
            ),  # Graceful shutdown
            backlog=2048,  # Increased socket backlog for better handling of concurrent connections
            loop="asyncio",  # Use asyncio event loop for better performance
        )

        server = uvicorn.Server(config)
        server.run()

    except Exception as e:
        logger.error(f"Error starting the server: {str(e)}", exc_info=True)
        sys.exit(1)


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Comprehensive health check of the service"""
    db_connected = check_db_connection()

    status = "healthy" if api_loaded and db_connected else "degraded"
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("VOYAGE_API_KEY"):
        status = "degraded"

    return {
        "status": status,
        "components": {
            "api": {
                "status": "up" if api_loaded else "down",
                "details": (
                    "API router loaded successfully"
                    if api_loaded
                    else "API router failed to load"
                ),
            },
            "database": {
                "status": "up" if db_connected else "down",
                "type": "ChromaDB",
                "persist_dir": os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db"),
            },
            "services": {
                "voyage": {
                    "status": (
                        "configured"
                        if os.environ.get("VOYAGE_API_KEY")
                        else "not configured"
                    )
                },
                "openai": {
                    "status": (
                        "configured"
                        if os.environ.get("OPENAI_API_KEY")
                        else "not configured"
                    )
                },
            },
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "protobuf_implementation": os.environ.get(
                "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "unknown"
            ),
        },
    }


@app.get("/chat", response_class=HTMLResponse)
def chat_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8" />
        <title>VOX AI Chat</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <style>
            body {
                margin: 0;
                height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(135deg,#667eea 0%, #764ba2 100%);
                font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
            }
            .chat-container {
                display: flex;
                flex-direction: column;
                background: #fff;
                border-radius: 16px;
                box-shadow: 0 8px 32px rgba(0,0,0,.2);
                width: 100%;
                max-width: 500px;
                height: 80vh;
                overflow: hidden;
            }
            .chat-header {
                padding: 16px;
                background: #667eea;
                color: #fff;
                font-size: 20px;
                font-weight: bold;
                text-align: center;
            }
            .chat-messages {
                flex: 1;
                padding: 16px;
                overflow-y: auto;
                background: #f9f9f9;
            }
            .message {
                max-width: 75%;
                margin-bottom: 12px;
                padding: 12px 16px;
                border-radius: 16px;
                font-size: 15px;
                line-height: 1.4;
            }
            .message.user {
                background: #667eea;
                color: white;
                margin-left: auto;
                border-bottom-right-radius: 4px;
            }
            .message.ai {
                background: #e5e5ea;
                color: #333;
                margin-right: auto;
                border-bottom-left-radius: 4px;
            }
            .chat-input {
                display: flex;
                border-top: 1px solid #ddd;
            }
            .chat-input input {
                flex: 1;
                border: none;
                padding: 14px;
                font-size: 15px;
                outline: none;
            }
            .chat-input button {
                background: #667eea;
                color: white;
                border: none;
                padding: 14px 20px;
                font-weight: bold;
                cursor: pointer;
                transition: background 0.2s;
            }
            .chat-input button:hover {
                background: #5a67d8;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">VOX AI Chat</div>
            <div class="chat-messages" id="messages">
                <div class="message ai">Hello 👋 I am your VOX assistant. How can I help?</div>
            </div>
            <form class="chat-input" onsubmit="sendMessage(event)">
                <input type="text" id="userInput" placeholder="Type your message..." autocomplete="off" />
                <button type="submit">Send</button>
            </form>
        </div>

        <script>
    const messages = document.getElementById("messages");
    const userInput = document.getElementById("userInput");

    // Open WebSocket connection
    const ws = new WebSocket("ws://localhost:4000/ai/conversation/chat"); // adjust host if needed

    ws.onopen = () => {
        console.log("✅ Connected to WebSocket server");
    };

    ws.onmessage = (event) => {
        appendMessage("🤖 " + event.data, "ai");
    };

    ws.onclose = () => {
        appendMessage("⚠️ Disconnected from server", "ai");
    };

    function appendMessage(text, sender) {
        const msg = document.createElement("div");
        msg.classList.add("message", sender);
        msg.textContent = text;
        messages.appendChild(msg);
        messages.scrollTop = messages.scrollHeight;
    }

  
function sendMessage(event) {
    event.preventDefault();
    const text = userInput.value.trim();
    if (!text) return;

    appendMessage(text, "user");
    ws.send(text); // 🔑 goes to backend
    userInput.value = "";
}
</script>

    </body>
    </html>
    """


if __name__ == "__main__":
    run_server()
