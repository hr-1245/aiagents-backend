from typing import Dict, Any, List, Optional
import uuid
import json
from datetime import datetime, timezone
from langchain_openai import ChatOpenAI
import logging

from ..models.models import CustomAIAgent, CreateAgentRequest, UpdateAgentRequest
from ..config import get_llm_config, get_dynamic_llm_config, get_humanlike_behavior_instructions
from ..services.supabase_service import SupabaseService

logger = logging.getLogger(__name__)

class CustomAgentService:
    """Service for managing custom AI agents."""
    
    def __init__(self):
        # Initialize Supabase service for database operations
        self.supabase_service = SupabaseService()
        
        # System user UUID - should match the system user in your database
        self.system_user_uuid = "00000000-0000-0000-0000-000000000000"
        
        logger.info("CustomAgentService initialized with database-only storage")
    
    def _convert_agent_id_to_uuid(self, agent_id: str) -> str:
        """Convert agent ID to UUID format - just return as is since frontend sends actual IDs."""
        return agent_id
    
    def _is_valid_uuid(self, uuid_string: str) -> bool:
        """Check if a string is a valid UUID."""
        try:
            uuid.UUID(uuid_string)
            return True
        except ValueError:
            return False
    
    async def get_agent(self, agent_id: str, user_id: str) -> Optional[CustomAIAgent]:
        """Get an agent by ID from database."""
        
        try:
            # Use the agent ID exactly as provided (no conversion)
            actual_agent_id = agent_id
            actual_user_id = user_id
            
            # Query database for the agent
            agent_data = await self.supabase_service.get_ai_agent(actual_agent_id, actual_user_id)
            if agent_data:
                # Transform to CustomAIAgent object
                agent = CustomAIAgent(**agent_data)
                logger.info(f"Successfully retrieved agent {agent_id} from database for user {user_id}")
                return agent
        except Exception as e:
            logger.warning(f"Error fetching agent from database: {e}")
        
        logger.warning(f"Agent {agent_id} not found in database for user {user_id}")
        return None
    
    async def create_agent(self, request: CreateAgentRequest) -> CustomAIAgent:
        """Create a new custom AI agent."""
        agent_id = str(uuid.uuid4())
        
        # Build the complete custom prompt from structured components
        custom_prompt = self._build_custom_prompt(
            personality=request.personality,
            intent=request.intent,
            additional_information=request.additionalInformation,
            agent_type=request.agentType
        )
        
        # Validate custom prompt to ensure it doesn't override core intelligence
        validated_prompt = self._validate_custom_prompt(custom_prompt)
        
        # Enhance with core instructions to maintain functionality
        enhanced_prompt = self._enhance_custom_prompt(validated_prompt, request.agentType)
        
        agent = CustomAIAgent(
            id=agent_id,
            userId=request.userId,
            name=request.name,
            description=request.description,
            agentType=request.agentType,
            personality=request.personality,
            intent=request.intent,
            additionalInformation=request.additionalInformation,
            variables=request.variables or {},
            customPrompt=enhanced_prompt,
            channels=request.channels or [], 
        )
        
        # Store in Supabase
        try:
            agent_dict = agent.model_dump()
            created_agent_data = await self.supabase_service.create_ai_agent(agent_dict)
            if created_agent_data:
                # Return the agent with the actual database ID
                created_agent = CustomAIAgent(**created_agent_data)
                logger.info(f"Successfully created custom agent {created_agent.id} for user {request.userId} in Supabase")
                return created_agent
            else:
                logger.error(f"Failed to store agent in Supabase for agent {agent_id}")
                raise RuntimeError("Failed to create agent in database")
        except Exception as e:
            logger.error(f"Error storing agent in Supabase: {e}")
            raise RuntimeError(f"Failed to create agent: {str(e)}")
    
    def _build_custom_prompt(self, personality: str, intent: str, additional_information: Optional[str], agent_type: str) -> str:
        """Build a custom prompt from structured components."""
        prompt_parts = []
        
        if personality:
            prompt_parts.append(f"Personality:\n{personality}")
        
        if intent:
            prompt_parts.append(f"Intent:\n{intent}")
        
        if additional_information:
            prompt_parts.append(f"Additional Information:\n{additional_information}")
        
        return "\n\n".join(prompt_parts)
    
    def _enhance_custom_prompt(self, custom_prompt: str, agent_type: str) -> str:
        """Enhance custom prompt to maintain core functionality."""
        
        # Core instructions that must be preserved
        core_instructions = {
            "query": """
IMPORTANT: You are responding directly to customers. Be helpful, accurate, and conversational.

CONVERSATION CONTEXT:
{context}

CUSTOMER'S QUESTION:
"{query}"

CORE INSTRUCTIONS:
- You have access to MULTIPLE SOURCES: conversation context, knowledge bases, documents, and contextual data - LEVERAGE ALL OF THEM
- When answering ANY query, draw from ALL available sources: conversation history, knowledge bases, document repositories, and contextual information
- CROSS-REFERENCE information across multiple knowledge bases to provide comprehensive, well-informed responses
- When customers ask about conversations, topics, products, services, or any subject, synthesize information from ALL available sources
- Never say "I don't know" when you have ANY available information sources - always provide the most comprehensive answer possible
- Use conversation context to understand intent, then enhance with knowledge base information for complete responses
- Reference specific details from conversations AND supplement with relevant knowledge base content
- For complex queries, combine insights from multiple knowledge sources to provide thorough, authoritative answers
- Always prioritize accuracy by cross-referencing multiple sources when available
- Be proactive in suggesting related information from knowledge bases that might be helpful to the customer

""",
            "suggestions": """
IMPORTANT: Generate exactly 3 follow-up messages for customer service representatives that leverage ALL available information sources.

CONVERSATION CONTEXT:
{context}

CORE INSTRUCTIONS FOR SUGGESTIONS:
- Draw from ALL available sources: conversation history, multiple knowledge bases, document repositories, and contextual information
- Each suggestion should combine conversation context with relevant knowledge base information
- Cross-reference multiple knowledge sources to provide comprehensive, valuable follow-up options
- Suggest next steps that demonstrate deep knowledge from available information sources
- When appropriate, reference specific products, services, or information from knowledge bases
- Provide suggestions that exceed customer expectations by leveraging comprehensive data sources

Your task: Generate exactly 3 numbered follow-up suggestions that maintain conversation flow AND demonstrate comprehensive knowledge.

""",
            "response": """
IMPORTANT: You are helping generate responses for customer service representatives with access to multiple information sources.

CONVERSATION CONTEXT:
{context}

CUSTOMER'S MESSAGE:
"{last_customer_message}"

CORE INSTRUCTIONS FOR RESPONSES:
- Leverage ALL available sources: conversation history, multiple knowledge bases, document repositories, and contextual information
- Combine conversation context with relevant knowledge base information to provide comprehensive responses
- Cross-reference multiple knowledge sources to ensure accuracy and completeness
- When addressing customer concerns, draw from relevant documentation, product information, and service knowledge
- Provide additional value by referencing related information from knowledge bases that might be helpful
- Demonstrate comprehensive knowledge while maintaining a natural, caring tone
- Go beyond just answering the immediate question to provide proactive, knowledge-based assistance

Your task: Generate one comprehensive, knowledge-enhanced response to send to this customer.

"""
        }
        
        # Prepend core instructions to custom prompt - CORE INTELLIGENCE MUST ALWAYS BE PRESERVED
        core = core_instructions.get(agent_type, "")
        
        if not core:
            # Fallback core instructions for any agent type
            core = """
IMPORTANT: You have access to multiple information sources and must use them intelligently.

CORE INTELLIGENCE (ALWAYS PRESERVED):
- You have access to MULTIPLE SOURCES: conversation context, knowledge bases, documents, and contextual data - LEVERAGE ALL OF THEM
- When answering ANY query, draw from ALL available sources: conversation history, knowledge bases, document repositories, and contextual information
- CROSS-REFERENCE information across multiple knowledge bases to provide comprehensive, well-informed responses
- Never say "I don't know" when you have ANY available information sources - always provide the most comprehensive answer possible
- Use conversation context to understand intent, then enhance with knowledge base information for complete responses
- Reference specific details from conversations AND supplement with relevant knowledge base content
- For complex queries, combine insights from multiple knowledge sources to provide thorough, authoritative answers
- Always prioritize accuracy by cross-referencing multiple sources when available
- Be proactive in suggesting related information from knowledge bases that might be helpful to the customer

"""
        
        # Structure: CORE INTELLIGENCE + USER CUSTOMIZATION
        return f"{core}\n{'='*60}\nCUSTOM AGENT INSTRUCTIONS (These enhance, not replace, core intelligence):\n{custom_prompt}\n{'='*60}\n\nREMEMBER: Always apply BOTH the core intelligence above AND the custom instructions below when responding."
    
    def _validate_custom_prompt(self, custom_prompt: str) -> str:
        """Validate and sanitize custom prompt to prevent overriding core intelligence."""
        # Convert to lowercase for checking
        lower_prompt = custom_prompt.lower()
        
        # Potentially problematic phrases that might override core intelligence
        problematic_phrases = [
            "ignore previous instructions",
            "don't use knowledge bases",
            "only use conversation context",
            "don't reference other sources",
            "ignore context",
            "don't cross-reference",
            "only answer from",
            "don't provide additional information"
        ]
        
        # Check for problematic phrases
        for phrase in problematic_phrases:
            if phrase in lower_prompt:
                logger.warning(f"Custom prompt contains potentially problematic phrase: '{phrase}'. Core intelligence will still be preserved.")
        
        # Add a protective instruction if the prompt seems to be trying to limit functionality
        if any(phrase in lower_prompt for phrase in ["only", "don't", "ignore", "limit to", "restrict to"]):
            custom_prompt += "\n\nNOTE: These custom instructions work alongside the core intelligence system and should not override the ability to use multiple information sources comprehensively."
        
        return custom_prompt
    
    def _apply_variables(self, text: str, variables: Dict[str, str]) -> str:
        """Apply dynamic variables to text using {{variable_name}} syntax."""
        result = text
        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"  # {{key}} format
            result = result.replace(placeholder, value)
        return result
    
    async def list_agents(self, user_id: str, agent_type: Optional[str] = None) -> List[CustomAIAgent]:
        """List agents for a user."""
        agents = []
        
        # Get agents from Supabase (user-created agents)
        try:
            agent_type_int = None
            if agent_type:
                agent_type_int = self.supabase_service._map_agent_type_to_db(agent_type)
            
            supabase_agents_data = await self.supabase_service.list_ai_agents(user_id, agent_type_int)
            for agent_data in supabase_agents_data:
                try:
                    agent = CustomAIAgent(**agent_data)
                    agents.append(agent)
                except Exception as e:
                    logger.warning(f"Error converting Supabase agent data to CustomAIAgent: {e}")
        except Exception as e:
            logger.warning(f"Error fetching agents from Supabase: {e}")
        
        # Remove duplicates (in case an agent exists in both stores)
        seen_ids = set()
        unique_agents = []
        for agent in agents:
            if agent.id not in seen_ids:
                unique_agents.append(agent)
                seen_ids.add(agent.id)
        
        return sorted(unique_agents, key=lambda x: x.createdAt, reverse=True)
    
    async def update_agent(self, agent_id: str, user_id: str, request: UpdateAgentRequest) -> Optional[CustomAIAgent]:
        """Update an existing agent."""
        agent = await self.get_agent(agent_id, user_id)
        if not agent or agent.userId != user_id:  # Can't update system agents
            return None
        
        # Update fields
        if request.channels is not None:
            agent.channels = request.channels
        if request.name is not None:
            agent.name = request.name
        if request.description is not None:
            agent.description = request.description
        if request.personality is not None:
            agent.personality = request.personality
        if request.intent is not None:
            agent.intent = request.intent
        if request.additionalInformation is not None:
            agent.additionalInformation = request.additionalInformation
        if request.variables is not None:
            agent.variables = request.variables
        if request.isActive is not None:
            agent.isActive = request.isActive
        
        # Rebuild custom prompt if any component changed
        if any([request.personality, request.intent, request.additionalInformation]):
            custom_prompt = self._build_custom_prompt(
                personality=agent.personality,
                intent=agent.intent,
                additional_information=agent.additionalInformation,
                agent_type=agent.agentType
            )
            # Validate custom prompt to ensure it doesn't override core intelligence
            validated_prompt = self._validate_custom_prompt(custom_prompt)
            agent.customPrompt = self._enhance_custom_prompt(validated_prompt, agent.agentType)
        
        agent.updatedAt = datetime.now(timezone.utc)
        
        # Try to update in Supabase
        try:
            if agent.userId != self.system_user_uuid:  # Don't update system agents in Supabase
                agent_dict = agent.model_dump()
                updated_agent_data = await self.supabase_service.update_ai_agent(agent_id, user_id, agent_dict)
                if updated_agent_data:
                    updated_agent = CustomAIAgent(**updated_agent_data)
                    logger.info(f"Successfully updated agent {agent_id} in Supabase for user {user_id}")
                    return updated_agent
                else:
                    logger.warning(f"Failed to update agent in Supabase for agent {agent_id}")
                    return None
            else:
                logger.warning(f"Cannot update system agent {agent_id}")
                return None
        except Exception as e:
            logger.error(f"Error updating agent in Supabase: {e}")
            return None
    
    async def delete_agent(self, agent_id: str, user_id: str) -> bool:
        """Delete an agent."""
        agent = await self.get_agent(agent_id, user_id)
        if not agent or agent.userId != user_id:  # Can't delete system agents
            return False
        
        # Try to delete from Supabase
        try:
            if agent.userId != self.system_user_uuid:  # Don't delete system agents from Supabase
                success = await self.supabase_service.delete_ai_agent(agent_id, user_id)
                if success:
                    logger.info(f"Successfully deleted agent {agent_id} from Supabase for user {user_id}")
                    return True
                else:
                    logger.warning(f"Failed to delete agent from Supabase for agent {agent_id}")
                    return False
            else:
                logger.warning(f"Cannot delete system agent {agent_id}")
                return False
        except Exception as e:
            logger.error(f"Error deleting agent from Supabase: {e}")
            return False
    
    async def execute_agent(self, agent_id: str, user_id: str, context: Dict[str, Any], 
                           temperature: float = None, model: str = None, 
                           humanlike_behavior: bool = False) -> str:
        """Execute a custom agent with given context and AI parameters."""
        agent = await self.get_agent(agent_id, user_id)
        if not agent or not agent.isActive:
            raise ValueError(f"Agent {agent_id} not found or inactive")
        
        try:
            # Get dynamic LLM config with custom parameters
            llm_config = get_dynamic_llm_config(temperature=temperature, model=model)
            llm = ChatOpenAI(**llm_config)
            
            # Apply dynamic variables to the prompt
            prompt_with_variables = self._apply_variables(agent.customPrompt, agent.variables or {})
            
            # Format the prompt with context
            formatted_prompt = prompt_with_variables.format(**context)
            
            # Add human-like behavior instructions if requested
            if humanlike_behavior:
                formatted_prompt += f"\n\n{get_humanlike_behavior_instructions()}"
            
            result = await llm.ainvoke(formatted_prompt)
            return result.content or ""
            
        except Exception as e:
            logger.error(f"Error executing agent {agent_id}: {e}")
            raise RuntimeError(f"Agent execution failed: {str(e)}")

# Global service instance
custom_agent_service = CustomAgentService() 