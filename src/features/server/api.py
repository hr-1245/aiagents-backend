from fastapi import APIRouter

# Import all route modules
from .routes import training, query, summary, suggestions, management, agents, websocket

# Create the main router for the conversation API
router = APIRouter()

# Include all route modules with their respective prefixes
router.include_router(training.router, tags=["Training"])
router.include_router(query.router, tags=["Query"])
router.include_router(summary.router, tags=["Summary"])
router.include_router(suggestions.router, tags=["Suggestions"])
router.include_router(management.router, tags=["Management"])
router.include_router(agents.router, tags=["Custom Agents"]) 
# ✅ Add WebSocket route (no prefix needed, it’s already /ws/train inside websocket.py)
router.include_router(websocket.router, tags=["WebSocket"])