from fastapi import APIRouter

# from sockets import sio_app

# Import all route modules
from src.features.server.sockets.sockets import router as socket_router
from .routes import (
    training,
    query,
    summary,
    suggestions,
    management,
    agents,
    knowledgebase,
)

# Create the main router for the conversation API
router = APIRouter()

# Include all route modules with their respective prefixes
router.include_router(knowledgebase.router, tags=["KnowledgeBase"])
router.include_router(training.router, tags=["Training"])
router.include_router(query.router, tags=["Query"])
router.include_router(summary.router, tags=["Summary"])
router.include_router(suggestions.router, tags=["Suggestions"])
router.include_router(management.router, tags=["Management"])
router.include_router(agents.router, tags=["Custom Agents"])
router.include_router(socket_router, tags=["Sockets"])
