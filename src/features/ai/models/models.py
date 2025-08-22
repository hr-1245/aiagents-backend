from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator
from langchain_core.documents import Document

def validate_temperature(value: float) -> float:
    """Validate temperature: must be 0, 1, or between 0.1-0.9"""
    if value is None:
        return value
    if value == 0.0 or value == 1.0:
        return value
    if 0.1 <= value <= 0.9:
        return value
    raise ValueError("Temperature must be 0, 1, or between 0.1-0.9")

# Pydantic model for chat messages
class Message(BaseModel):
    id: str  # Unique message identifier
    body: Optional[str] = None  # Message content
    direction: Optional[str] = None  # 'inbound' or 'outbound' - made optional
    dateAdded: Optional[Union[str, datetime]] = None  # Timestamp of message creation
    messageType: Optional[str] = None  # Type of message (e.g., text, image)
    contentType: Optional[str] = "text/plain"  # MIME type of content
    status: Optional[str] = None  # Message status (e.g., sent, delivered)
    type: Optional[int] = None  # Legacy type field
    role: Optional[str] = None  # Role of sender (e.g., customer, system user)
    conversationId: Optional[str] = None  # Conversation identifier
    knowledgebaseId: Optional[str] = None  # Knowledgebase identifier
    locationId: Optional[str] = None  # Location identifier
    contactId: Optional[str] = None  # Contact identifier
    userId: Optional[str] = None  # User identifier
    source: Optional[str] = None  # Source of message (e.g., conversation, web)

    @field_validator('direction')
    def validate_direction(cls, v):
        # Ensure direction is either 'inbound' or 'outbound' or None
        if v and v not in ['inbound', 'outbound']:
            raise ValueError("Direction must be 'inbound' or 'outbound'")
        return v

    def clean_data(self) -> dict:
        # Clean and normalize message data
        data = self.model_dump()
        cleaned = data.copy()  # Create a copy to avoid modifying original

        # Standardize ID field
        for id_field in ['id', 'messageId', 'uid']:
            if id_val := data.get(id_field):
                cleaned['id'] = id_val
                break

        # Standardize content field
        for field in ['body', 'content', 'text', 'message']:
            if content := data.get(field):
                if isinstance(content, str):
                    cleaned['body'] = content
                    break

        # Infer direction if not provided
        if not cleaned.get('direction'):
            role = cleaned.get('role', '').lower()
            if 'user' in role or 'customer' in role:
                cleaned['direction'] = 'inbound'
            elif 'assistant' in role or 'system' in role or 'ai' in role:
                cleaned['direction'] = 'outbound'
            else:
                cleaned['direction'] = 'outbound'  # Default direction

        # Infer role from direction if not provided
        if not cleaned.get('role'):
            direction = cleaned.get('direction')
            cleaned["role"] = "customer" if direction == "inbound" else "system user" if direction == "outbound" else "system notification"

        return cleaned

# Document class for vector store, supporting multiple data types
class VectorDocument:
    @staticmethod
    def from_message(message: Message, userId: str, data_type: str = "message", knowledgebaseId: Optional[str] = None) -> Document:
        # Initialize document with content and metadata
        content = message.body if message.body else ""
        date_added = message.dateAdded if isinstance(message.dateAdded, str) else (
            message.dateAdded.isoformat() if message.dateAdded else datetime.now(timezone.utc).isoformat()
        )
        kb_id = knowledgebaseId or message.knowledgebaseId or message.conversationId
        
        # Get direction or infer from role
        direction = message.direction
        if not direction:
            role = message.role or ""
            if "user" in role.lower() or "customer" in role.lower():
                direction = "inbound"
            else:
                direction = "outbound"
        
        metadata = {
            "userId": userId,
            "messageId": message.id,
            "direction": direction,
            "conversationId": message.conversationId or "",
            "knowledgebaseId": kb_id or "",
            "dateAdded": date_added,
            "role": message.role or ("user" if direction == "inbound" else "assistant"),
            "type": data_type  # Differentiate data types (message, web, pdf)
        }
        if message.locationId:
            metadata["locationId"] = message.locationId
        if message.messageType:
            metadata["messageType"] = message.messageType
        if message.contentType:
            metadata["contentType"] = message.contentType
        if message.source:
            metadata["source"] = message.source
            
        return Document(page_content=content, metadata=metadata)

# Request model for training conversations
class MessageRequest(BaseModel):
    userId: str
    conversationId: str
    messages: List[Message]
    messageCount: Optional[int] = None
    locationId: Optional[str] = None
    lastMessageId: Optional[str] = None
    knowledgebaseId: Optional[str] = None

# Response model for training operations
class MessageResponse(BaseModel):
    success: bool
    message: str
    data: Dict[str, Any] = Field(default_factory=lambda: {"conversationId": "", "messageCount": 0, "dateRange": {"start": "", "end": ""}})
    external: Optional[Dict[str, Any]] = None

# Agent info from frontend
class AgentInfo(BaseModel):
    id: str
    name: str
    prompt: str
    type: Optional[int] = None

# Request model for querying conversations
class QueryRequest(BaseModel):
    userId: str
    conversationId: Optional[str] = Field(default=None, description="Conversation ID - if not provided, queries knowledge bases only")
    query: str
    knowledgebaseId: Optional[str] = None
    locationId: Optional[str] = None
    limit: Optional[int] = 5
    filters: Optional[Dict[str, Any]] = None
    additionalKnowledgebaseIds: Optional[List[str]] = Field(default_factory=list, description="Additional knowledge bases to query")
    aiAgentId: Optional[str] = Field(default=None, description="Custom AI agent to use")
    agentInfo: Optional[AgentInfo] = Field(default=None, description="Agent information from frontend")
    context: Optional[str] = Field(default="", description="Additional context")
    customerInfo: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Customer information")
    recentMessages: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Recent messages")
    knowledgebaseIds: Optional[List[str]] = Field(default_factory=list, description="Knowledge base IDs")
    systemPrompt: Optional[str] = Field(default=None, description="System prompt override")
    
    # AI Configuration Parameters
    temperature: Optional[float] = Field(default=None, description="AI response creativity/randomness (0, 1, or 0.1-0.9)")
    model: Optional[str] = Field(default=None, description="AI model to use (e.g., 'gpt-4o', 'gpt-4-turbo')")
    humanlikeBehavior: Optional[bool] = Field(default=False, description="Add human-like behavior instructions to AI responses")
    
    @field_validator('temperature')
    def validate_temperature_field(cls, v):
        return validate_temperature(v)

# Response model for query operations
class QueryResponse(BaseModel):
    messages: List[Dict[str, Any]]
    total: int
    query: str
    answer: str
    suggestions: Optional[List[str]] = Field(default_factory=list, description="Follow-up suggestions")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Request model for summarizing conversations
class SummaryRequest(BaseModel):
    userId: str
    conversationId: str
    locationId: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    
    # AI Configuration Parameters
    temperature: Optional[float] = Field(default=None, description="AI response creativity/randomness (0, 1, or 0.1-0.9)")
    model: Optional[str] = Field(default=None, description="AI model to use (e.g., 'gpt-4o', 'gpt-4-turbo')")
    humanlikeBehavior: Optional[bool] = Field(default=False, description="Add human-like behavior instructions to AI responses")
    
    @field_validator('temperature')
    def validate_temperature_field(cls, v):
        return validate_temperature(v)

# Response model for summary operations
class ConversationSummary(BaseModel):
    success: bool
    summary: str
    metadata: Dict[str, Any]
    is_trained: Optional[bool] = Field(default=None, description="Whether the conversation has been trained")
    error_code: Optional[str] = Field(default=None, description="Error code for specific error types")
    recommendations: Optional[List[str]] = Field(default_factory=list, description="Recommendations for the user")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Response model for checking training status
class TrainingStatus(BaseModel):
    is_trained: bool
    last_updated: Optional[str] = None
    message_count: Optional[int] = None
    vector_count: Optional[int] = None

# Request model for generating suggestions
class SuggestionRequest(BaseModel):
    userId: str
    conversationId: Optional[str] = Field(default=None, description="Conversation ID - if not provided, queries knowledge bases only")
    query: Optional[str] = None
    context: Optional[str] = None
    knowledgebaseId: Optional[str] = None
    limit: Optional[int] = Field(default=3, description="Number of suggestions to generate (1-6)", ge=1, le=6)
    recentMessages: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Last 5 messages with full details and roles")
    additionalKnowledgebaseIds: Optional[List[str]] = Field(default_factory=list, description="Additional knowledge bases to query")
    aiAgentId: Optional[str] = Field(default=None, description="Custom AI agent to use")
    
    # AI Configuration Parameters
    temperature: Optional[float] = Field(default=None, description="AI response creativity/randomness (0, 1, or 0.1-0.9)")
    model: Optional[str] = Field(default=None, description="AI model to use (e.g., 'gpt-4o', 'gpt-4-turbo')")
    humanlikeBehavior: Optional[bool] = Field(default=False, description="Add human-like behavior instructions to AI responses")
    
    @field_validator('temperature')
    def validate_temperature_field(cls, v):
        return validate_temperature(v)

# Response model for suggestions
class SuggestionResponse(BaseModel):
    suggestions: List[str]
    total: int
    conversationId: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Request model for generating response suggestions
class ResponseSuggestionRequest(BaseModel):
    userId: str
    conversationId: Optional[str] = Field(default=None, description="Conversation ID - if not provided, queries knowledge bases only")
    knowledgebaseId: Optional[str] = None
    context: Optional[str] = None
    lastCustomerMessage: Optional[str] = None
    recentMessages: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Last 5 messages with full details and roles")
    autopilot: Optional[bool] = Field(default=False, description="If true, automatically select and return the best response for autopilot mode")
    additionalKnowledgebaseIds: Optional[List[str]] = Field(default_factory=list, description="Additional knowledge bases to query")
    aiAgentId: Optional[str] = Field(default=None, description="Custom AI agent to use")
    
    # Additional autopilot parameters from Next.js
    systemPrompt: Optional[str] = Field(default=None, description="Agent-specific system prompt/instructions")
    agentName: Optional[str] = Field(default=None, description="Agent's name for personalization")
    customerInfo: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Customer information for personalization")
    
    # AI Configuration Parameters
    temperature: Optional[float] = Field(default=0.7, description="AI response creativity/randomness (0, 1, or 0.1-0.9)")
    model: Optional[str] = Field(default="gpt-4o-mini", description="AI model to use (e.g., 'gpt-4o', 'gpt-4-turbo', 'gpt-4o-mini')")
    humanlikeBehavior: Optional[bool] = Field(default=False, description="Add human-like behavior instructions to AI responses")
    
    @field_validator('temperature')
    def validate_temperature_field(cls, v):
        return validate_temperature(v)

# Response model for response suggestions
class ResponseSuggestionResponse(BaseModel):
    response_suggestion: str = Field(description="The single best response suggestion")
    response_suggestions: List[str] = Field(description="List of response suggestions (for backward compatibility)")
    total: int = Field(description="Total number of suggestions")
    conversationId: str
    autopilot_response: Optional[str] = Field(default=None, description="The selected response for autopilot mode")
    confidence_score: Optional[float] = Field(default=None, description="Confidence score for autopilot response (0.0-1.0)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# ============================================================================
# CUSTOM AI AGENTS MODELS
# ============================================================================

class CustomAIAgent(BaseModel):
    """Model for custom AI agents with structured personality, intent, and additional information."""
    id: str = Field(description="Unique agent identifier")
    userId: str = Field(description="Owner of the agent")
    name: str = Field(description="Human-readable agent name")
    description: Optional[str] = Field(default=None, description="Agent description")
    agentType: str = Field(description="Type: 'query', 'suggestions', 'response'")
    
    # Structured prompt components
    personality: str = Field(description="Agent's personality and role definition")
    intent: str = Field(description="Agent's primary goal and purpose")
    additionalInformation: Optional[str] = Field(default=None, description="Guidelines, examples, rules, and constraints")
    
    # Dynamic variables for prompt templating
    variables: Optional[Dict[str, str]] = Field(default_factory=dict, description="Dynamic variables like business_name, etc.")
    
    # Legacy field for backward compatibility
    customPrompt: Optional[str] = Field(default=None, description="Generated combined prompt (auto-created)")
    
    isActive: bool = Field(default=True, description="Whether agent is active")
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updatedAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @field_validator('agentType')
    def validate_agent_type(cls, v):
        valid_types = ['query', 'suggestions', 'response']
        if v not in valid_types:
            error_msg = (
                f"Agent type must be one of: {valid_types}. "
                f"Each type has a specific purpose:\n"
                f"- 'query': Responds directly to customer questions\n"
                f"- 'suggestions': Generates exactly 3 follow-up suggestions for reps\n" 
                f"- 'response': Generates response options for customer messages\n"
                f"Received invalid type: '{v}'"
            )
            raise ValueError(error_msg)
        return v

class CreateAgentRequest(BaseModel):
    """Request to create a new custom AI agent with structured components."""
    userId: str
    name: str
    description: Optional[str] = None
    agentType: str
    
    # Required structured components
    personality: str = Field(description="Agent's personality and role (e.g., 'You are a bot for {{business_name}}, tasked to assist customers...')")
    intent: str = Field(description="Agent's primary goal (e.g., 'Your goal is to assist customers with their queries')")
    additionalInformation: Optional[str] = Field(default=None, description="Guidelines, examples, rules, conversation style, etc.")
    
    # Dynamic variables
    variables: Optional[Dict[str, str]] = Field(
        default_factory=dict, 
        description="Dynamic variables for templating (e.g., {'business_name': 'Acme Corp', 'support_email': 'help@acme.com'})"
    )

class UpdateAgentRequest(BaseModel):
    """Request to update an existing custom AI agent."""
    name: Optional[str] = None
    description: Optional[str] = None
    personality: Optional[str] = None
    intent: Optional[str] = None
    additionalInformation: Optional[str] = None
    variables: Optional[Dict[str, str]] = None
    isActive: Optional[bool] = None

class AgentResponse(BaseModel):
    """Response for agent operations."""
    success: bool
    message: str
    agent: Optional[CustomAIAgent] = None

class ListAgentsResponse(BaseModel):
    """Response for listing agents."""
    success: bool
    agents: List[CustomAIAgent]
    total: int

# ============================================================================
# ENHANCED REQUEST MODELS WITH CUSTOM AGENTS & MULTI-KB
# ============================================================================

class EnhancedQueryRequest(BaseModel):
    """Enhanced query request with custom agent and multi-KB support."""
    userId: str
    conversationId: Optional[str] = Field(default=None, description="Conversation ID - if not provided, queries knowledge bases only")
    query: str
    knowledgebaseId: Optional[str] = None
    additionalKnowledgebaseIds: Optional[List[str]] = Field(default_factory=list, description="Additional knowledge bases to query")
    aiAgentId: Optional[str] = Field(default=None, description="Custom AI agent to use")
    locationId: Optional[str] = None
    limit: Optional[int] = 5
    filters: Optional[Dict[str, Any]] = None
    
    # AI Configuration Parameters
    temperature: Optional[float] = Field(default=None, description="AI response creativity/randomness (0, 1, or 0.1-0.9)")
    model: Optional[str] = Field(default=None, description="AI model to use (e.g., 'gpt-4o', 'gpt-4-turbo')")
    humanlikeBehavior: Optional[bool] = Field(default=False, description="Add human-like behavior instructions to AI responses")
    
    @field_validator('temperature')
    def validate_temperature_field(cls, v):
        return validate_temperature(v)

class EnhancedSuggestionRequest(BaseModel):
    """Enhanced suggestion request with custom agent and multi-KB support."""
    userId: str
    conversationId: Optional[str] = Field(default=None, description="Conversation ID - if not provided, queries knowledge bases only")
    query: Optional[str] = None
    context: Optional[str] = None
    knowledgebaseId: Optional[str] = None
    additionalKnowledgebaseIds: Optional[List[str]] = Field(default_factory=list, description="Additional knowledge bases to query")
    aiAgentId: Optional[str] = Field(default=None, description="Custom AI agent to use")
    agentInfo: Optional[AgentInfo] = Field(default=None, description="Frontend agent information as fallback")
    limit: Optional[int] = Field(default=3, description="Number of suggestions to generate (1-6)", ge=1, le=6)
    recentMessages: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    
    # AI Configuration Parameters
    temperature: Optional[float] = Field(default=None, description="AI response creativity/randomness (0, 1, or 0.1-0.9)")
    model: Optional[str] = Field(default=None, description="AI model to use (e.g., 'gpt-4o', 'gpt-4-turbo')")
    humanlikeBehavior: Optional[bool] = Field(default=False, description="Add human-like behavior instructions to AI responses")
    
    @field_validator('temperature')
    def validate_temperature_field(cls, v):
        return validate_temperature(v)

class EnhancedResponseSuggestionRequest(BaseModel):
    """Enhanced response suggestion request with custom agent and multi-KB support."""
    userId: str
    conversationId: Optional[str] = Field(default=None, description="Conversation ID - if not provided, queries knowledge bases only")
    knowledgebaseId: Optional[str] = None
    additionalKnowledgebaseIds: Optional[List[str]] = Field(default_factory=list, description="Additional knowledge bases to query")
    aiAgentId: Optional[str] = Field(default=None, description="Custom AI agent to use")
    context: Optional[str] = None
    lastCustomerMessage: Optional[str] = None
    recentMessages: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    autopilot: Optional[bool] = Field(default=False)
    
    # Autopilot Configuration
    systemPrompt: Optional[str] = Field(default=None, description="System prompt for autopilot mode")
    agentName: Optional[str] = Field(default=None, description="Agent name for autopilot mode")
    customerInfo: Optional[Dict[str, Any]] = Field(default=None, description="Customer information for autopilot mode")
    
    # AI Configuration Parameters
    temperature: Optional[float] = Field(default=None, description="AI response creativity/randomness (0, 1, or 0.1-0.9)")
    model: Optional[str] = Field(default=None, description="AI model to use (e.g., 'gpt-4o', 'gpt-4-turbo')")
    humanlikeBehavior: Optional[bool] = Field(default=False, description="Add human-like behavior instructions to AI responses")
    
    @field_validator('temperature')
    def validate_temperature_field(cls, v):
        return validate_temperature(v)

# ============================================================================
# TRAINING DATA SOURCES MODELS
# ============================================================================

class WebsiteCrawlRequest(BaseModel):
    """Request to crawl a website for training data."""
    userId: str
    knowledgebaseId: str
    url: str
    maxPages: Optional[int] = Field(default=10, description="Maximum pages to crawl")
    maxDepth: Optional[int] = Field(default=2, description="Maximum crawl depth")
    includePatterns: Optional[List[str]] = Field(default_factory=list, description="URL patterns to include")
    excludePatterns: Optional[List[str]] = Field(default_factory=list, description="URL patterns to exclude")

class DocumentUploadRequest(BaseModel):
    """Request to upload documents for training."""
    userId: str
    knowledgebaseId: str
    fileName: str
    fileType: str = Field(description="File type: pdf, docx, txt, etc.")
    content: Optional[str] = Field(default=None, description="Text content if already extracted")

class FAQTrainingRequest(BaseModel):
    """Request to train with FAQ data."""
    userId: str
    knowledgebaseId: str
    faqs: List[Dict[str, str]] = Field(description="List of FAQ items with 'question' and 'answer' keys")

class TrainingResponse(BaseModel):
    """Response for training operations."""
    success: bool
    message: str
    documentsProcessed: int
    vectorsCreated: int
    knowledgebaseId: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CrawlStatus(BaseModel):
    """Status of website crawling operation."""
    crawlId: str
    status: str = Field(description="Status: pending, running, completed, failed")
    pagesProcessed: int = 0
    totalPages: int = 0
    documentsCreated: int = 0
    startedAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completedAt: Optional[datetime] = None
    error: Optional[str] = None

# ============================================================================
# SUPABASE FILE TRAINING MODELS
# ============================================================================

class SupabaseFileTrainingRequest(BaseModel):
    """Request to train with a file stored in Supabase."""
    userId: str
    knowledgebaseId: str
    fileId: str = Field(description="Supabase file ID/path")
    fileName: str = Field(description="Original filename")
    fileType: str = Field(description="File type: pdf, docx, txt, etc.")
    fileSize: Optional[int] = Field(default=None, description="File size in bytes")
    supabaseBucket: str = Field(description="Supabase storage bucket name")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional file metadata")

class SupabaseDocumentTrainingResponse(BaseModel):
    """Response for Supabase document training."""
    success: bool
    message: str
    fileId: str
    fileName: str
    documentsProcessed: int
    vectorsCreated: int
    knowledgebaseId: str
    trainingStatus: str = Field(description="Status: processing, completed, failed")
    processingTime: Optional[float] = Field(default=None, description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TrainingJobStatus(BaseModel):
    """Status of a training job."""
    jobId: str
    userId: str
    knowledgebaseId: str
    fileId: str
    fileName: str
    status: str = Field(description="Status: pending, processing, completed, failed")
    progress: float = Field(default=0.0, description="Progress percentage (0.0-1.0)")
    documentsProcessed: int = 0
    vectorsCreated: int = 0
    startedAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completedAt: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

# ============================================================================
# SIMPLIFIED TRAINING DATA SOURCES MODELS
# ============================================================================

class SimpleFileTrainingRequest(BaseModel):
    """Simplified request to train with a file stored in Supabase."""
    userId: str = Field(description="User ID")
    fileId: str = Field(description="File ID from Supabase (becomes knowledgebaseId)")

class SimpleFAQTrainingRequest(BaseModel):
    """Simplified request to train with FAQ data."""
    userId: str = Field(description="User ID")
    faqs: List[Dict[str, str]] = Field(description="List of FAQ items with 'question' and 'answer' keys")
    knowledgebaseId: Optional[str] = Field(default=None, description="Optional custom knowledgebase ID, auto-generated if not provided")

class SimpleTrainingResponse(BaseModel):
    """Simplified response for training operations."""
    success: bool
    message: str
    knowledgebaseId: str
    documentsProcessed: int
    vectorsCreated: int
    processingTime: Optional[float] = Field(default=None, description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SimpleTrainingJobStatus(BaseModel):
    """Simplified status of a training job."""
    jobId: str
    userId: str
    knowledgebaseId: str
    fileId: str
    status: str = Field(description="Status: pending, processing, completed, failed")
    progress: float = Field(default=0.0, description="Progress percentage (0.0-1.0)")
    documentsProcessed: int = 0
    vectorsCreated: int = 0
    startedAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completedAt: Optional[datetime] = None
    error: Optional[str] = None

# ============================================================================
# ORIGINAL TRAINING DATA SOURCES MODELS (KEPT FOR BACKWARD COMPATIBILITY)
# ============================================================================

# ============================================================================
# UPDATE REQUEST/RESPONSE MODELS
# ============================================================================

class MessageUpdate(BaseModel):
    """Model for updating a single message or knowledge base item."""
    id: str  # ID of the message/item to update
    body: Optional[str] = None  # New content
    role: Optional[str] = None  # New role
    messageType: Optional[str] = None  # New message type
    contentType: Optional[str] = None  # New content type
    source: Optional[str] = None  # New source
    status: Optional[str] = None  # New status
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata updates

class ConversationUpdateRequest(BaseModel):
    """Request to update conversation messages."""
    userId: str
    conversationId: str
    knowledgebaseId: Optional[str] = None
    updates: List[MessageUpdate]  # List of message updates to apply
    updateType: str = "partial"  # "partial" or "replace" - how to handle updates

class KnowledgeBaseUpdateRequest(BaseModel):
    """Request to update knowledge base content."""
    userId: str
    knowledgebaseId: str
    updates: List[MessageUpdate]  # List of content updates to apply
    updateType: str = "partial"  # "partial" or "replace" - how to handle updates
    contentType: str = "knowledge_base"  # Type of content being updated

class BulkUpdateRequest(BaseModel):
    """Request for bulk updates across multiple conversations/knowledge bases."""
    userId: str
    conversationUpdates: Optional[List[ConversationUpdateRequest]] = []
    knowledgeBaseUpdates: Optional[List[KnowledgeBaseUpdateRequest]] = []

class UpdateResponse(BaseModel):
    """Response for update operations."""
    success: bool
    message: str
    updatedCount: int
    failedCount: int
    errors: Optional[List[str]] = []
    updatedIds: Optional[List[str]] = []
    timestamp: datetime

class BulkUpdateResponse(BaseModel):
    """Response for bulk update operations."""
    success: bool
    message: str
    conversationResults: List[UpdateResponse]
    knowledgeBaseResults: List[UpdateResponse]
    totalUpdated: int
    totalFailed: int
    timestamp: datetime
