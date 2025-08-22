from fastapi import APIRouter, HTTPException, Path, Query, Depends
from typing import Optional

from ...ai.models.models import (
    CreateAgentRequest, UpdateAgentRequest, AgentResponse, 
    ListAgentsResponse, CustomAIAgent
)
from ...ai.agents.custom_agent_service import custom_agent_service
from ..config import logger
from ..security import authenticate, check_rate_limit

router = APIRouter()

@router.post("/agents", response_model=AgentResponse)
async def create_custom_agent(
    request: CreateAgentRequest,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit)
):
    """
    Create a new custom AI agent with structured components.
    
    Required Structure:
    - personality: Agent's role and character (e.g., "You are a bot for {{business_name}}, tasked to assist customers...")
    - intent: Agent's primary goal (e.g., "Your goal is to assist customers with their queries")
    - additionalInformation: Guidelines, examples, rules (optional)
    - variables: Dynamic variables like {{business_name}}, {{support_email}} (optional)
    
    Agent Types:
    - 'query': Responds directly to customer questions (preserves conversational behavior)
    - 'suggestions': Generates exactly 3 follow-up suggestions for reps (preserves 3-suggestion format)  
    - 'response': Generates response options for customer messages (preserves response structure)
    
    Example Request:
    {
        "userId": "user123",
        "name": "Business Support Bot",
        "agentType": "query",
        "personality": "You are a bot for {{business_name}}, tasked to assist customers...",
        "intent": "Your goal is to assist customers with their queries",
        "additionalInformation": "Conversation Guidelines:\\n* Maintain casual tone\\n* Keep responses 20-25 words",
        "variables": {"business_name": "Acme Corp", "support_email": "help@acme.com"}
    }
    
    Note: Core instructions are automatically added to maintain functionality.
    """
    try:
        agent = await custom_agent_service.create_agent(request)
        return AgentResponse(
            success=True,
            message="Custom agent created successfully",
            agent=agent
        )
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")

@router.get("/agents", response_model=ListAgentsResponse)
async def list_agents(
    userId: str = Query(description="User ID to list agents for"),
    agentType: Optional[str] = Query(None, description="Filter by agent type: query, suggestions, response"),
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit)
):
    """List all agents available to a user."""
    try:
        agents = await custom_agent_service.list_agents(userId, agentType)
        return ListAgentsResponse(
            success=True,
            agents=agents,
            total=len(agents)
        )
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")

@router.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str = Path(description="Agent ID"),
    userId: str = Query(description="User ID"),
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Get a specific agent by ID."""
    try:
        agent = await custom_agent_service.get_agent(agent_id, userId)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return AgentResponse(
            success=True,
            message="Agent retrieved successfully",
            agent=agent
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent: {str(e)}")

@router.put("/agents/{agent_id}", response_model=AgentResponse)
async def update_agent(
    request: UpdateAgentRequest,
    agent_id: str = Path(description="Agent ID"),
    userId: str = Query(description="User ID"),
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Update an existing custom agent."""
    try:
        agent = await custom_agent_service.update_agent(agent_id, userId, request)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found or access denied")
        
        return AgentResponse(
            success=True,
            message="Agent updated successfully",
            agent=agent
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update agent: {str(e)}")

@router.delete("/agents/{agent_id}")
async def delete_agent(
    agent_id: str = Path(description="Agent ID"),
    userId: str = Query(description="User ID"),
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Delete a custom agent."""
    try:
        success = await custom_agent_service.delete_agent(agent_id, userId)
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found or access denied")
        
        return {"success": True, "message": "Agent deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete agent: {str(e)}")

@router.post("/agents/{agent_id}/test")
async def test_agent(
    test_data: dict,
    agent_id: str = Path(description="Agent ID"),
    userId: str = Query(description="User ID"),
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Test a custom agent with sample data."""
    try:
        # Validate test data based on agent type
        agent = await custom_agent_service.get_agent(agent_id, userId)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Prepare context based on agent type
        if agent.agentType == "query":
            context = {
                "context": test_data.get("context", "Sample conversation context"),
                "query": test_data.get("query", "Sample customer question")
            }
        elif agent.agentType == "suggestions":
            context = {
                "context": test_data.get("context", "Sample conversation context"),
                "limit": test_data.get("limit", 3)
            }
        elif agent.agentType == "response":
            context = {
                "context": test_data.get("context", "Sample conversation context"),
                "last_customer_message": test_data.get("last_customer_message", "Sample customer message")
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid agent type")
        
        result = await custom_agent_service.execute_agent(agent_id, userId, context)
        
        return {
            "success": True,
            "message": "Agent test completed",
            "result": result,
            "agent_type": agent.agentType
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to test agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to test agent: {str(e)}") 