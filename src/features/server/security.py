"""
Security module for VOX Backend API
Handles authentication, authorization, and rate limiting
"""

import os
import time
import hmac
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
import asyncio
from functools import lru_cache

logger = logging.getLogger(__name__)

class SecurityConfig:
    """Security configuration and constants"""
    
    # API Secret Key - should be set in environment
    API_SECRET_KEY = os.getenv("VOX_API_SECRET_KEY", "vox-dev-secret-key-change-in-production")
    
    # JWT-like token settings
    ACCESS_TOKEN_EXPIRE_MINUTES = 60
    ALGORITHM = "HS256"
    
    # Rate limiting - enhanced for concurrent requests
    RATE_LIMIT_REQUESTS = 500  # Increased for better performance
    RATE_LIMIT_WINDOW = 60     # seconds
    
    # API Key settings - Updated to user specifications
    API_KEY_HEADER = "X-API-Key"  # Changed from X-VOX-API-KEY
    API_KEY_PREFIX = "vox_"
    
    # Allowed origins for CORS (you'll configure this based on your frontend)
    ALLOWED_ORIGINS = [
        "http://localhost:3000",      # React dev server
        "http://localhost:3001",      # Alternative React port
        "http://127.0.0.1:3000",      # Alternative localhost
        "https://your-frontend-domain.com",  # Your production frontend
        # Add your actual frontend URLs here
    ]
    
    # Request signature settings (for extra security)
    SIGNATURE_HEADER = "X-VOX-Signature"
    TIMESTAMP_HEADER = "X-VOX-Timestamp"
    SIGNATURE_TOLERANCE = 300  # 5 minutes tolerance for timestamp

class APIKeyAuth:
    """API Key based authentication with performance optimizations"""
    
    def __init__(self):
        self.valid_api_keys = self._load_api_keys()
        self.bearer_scheme = HTTPBearer(auto_error=False)
        self._key_cache = {}  # Cache for faster lookups
    
    @lru_cache(maxsize=128)
    def _load_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load API keys from environment or database - cached for performance"""
        keys = {}
        
        # Primary API key from user's specified environment variable
        primary_key = os.getenv("VOX_API_X_KEY")
        if primary_key:
            keys[primary_key] = {
                "name": "frontend",
                "permissions": ["read", "write"],
                "rate_limit": 2000,  # Higher limit for frontend
                "created_at": datetime.now(timezone.utc),
                "last_used": None
            }
        
        # Admin API key (backup)
        admin_key = os.getenv("VOX_ADMIN_API_KEY")
        if admin_key:
            keys[admin_key] = {
                "name": "admin",
                "permissions": ["read", "write", "admin"],
                "rate_limit": 10000,  # Very high limit for admin
                "created_at": datetime.now(timezone.utc),
                "last_used": None
            }
        
        # Generate development key if no keys are configured
        if not keys:
            dev_key = f"{SecurityConfig.API_KEY_PREFIX}dev_{secrets.token_urlsafe(32)}"
            keys[dev_key] = {
                "name": "development",
                "permissions": ["read", "write"],
                "rate_limit": 100,
                "created_at": datetime.now(timezone.utc),
                "last_used": None
            }
            logger.warning(f"Generated development API key: {dev_key}")
            logger.warning("Set VOX_API_X_KEY in production!")
        
        return keys
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return key info - optimized for concurrent requests"""
        if api_key in self.valid_api_keys:
            # Update last used timestamp (non-blocking)
            self.valid_api_keys[api_key]["last_used"] = datetime.now(timezone.utc)
            return self.valid_api_keys[api_key]
        return None
    
    async def authenticate_request(self, request: Request) -> Dict[str, Any]:
        """Authenticate incoming request using API key - async optimized"""
        
        # Check for API key in the user's specified header
        api_key = request.headers.get(SecurityConfig.API_KEY_HEADER)
        
        # Also check for Bearer token format (backward compatibility)
        if not api_key:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                api_key = auth_header[7:]  # Remove "Bearer " prefix
        
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail=f"API key required. Include '{SecurityConfig.API_KEY_HEADER}' header with value from VOX_API_X_KEY environment variable",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Verify API key (optimized for concurrent access)
        key_info = self.verify_api_key(api_key)
        if not key_info:
            logger.warning(f"Invalid API key attempted from {request.client.host if request.client else 'unknown'}")
            raise HTTPException(
                status_code=401,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Add authentication info to request state
        request.state.api_key_info = key_info
        request.state.authenticated = True
        
        logger.info(f"Authenticated request from {key_info['name']} ({request.client.host if request.client else 'unknown'})")
        return key_info

class RequestSignatureAuth:
    """Request signature authentication for extra security"""
    
    @staticmethod
    def generate_signature(secret_key: str, timestamp: str, body: str) -> str:
        """Generate HMAC signature for request"""
        message = f"{timestamp}.{body}"
        signature = hmac.new(
            secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    @staticmethod
    def verify_signature(request: Request, secret_key: str) -> bool:
        """Verify request signature"""
        signature = request.headers.get(SecurityConfig.SIGNATURE_HEADER)
        timestamp = request.headers.get(SecurityConfig.TIMESTAMP_HEADER)
        
        if not signature or not timestamp:
            return False
        
        # Check timestamp tolerance
        try:
            request_time = int(timestamp)
            current_time = int(time.time())
            if abs(current_time - request_time) > SecurityConfig.SIGNATURE_TOLERANCE:
                logger.warning(f"Request timestamp out of tolerance: {abs(current_time - request_time)}s")
                return False
        except ValueError:
            return False
        
        # Get request body (you'll need to implement this based on your needs)
        # For now, we'll skip body verification for GET requests
        body = ""  # You can enhance this to include request body
        
        expected_signature = RequestSignatureAuth.generate_signature(secret_key, timestamp, body)
        
        return hmac.compare_digest(signature, expected_signature)

class RateLimiter:
    """Rate limiter with optimizations for concurrent requests"""
    
    def __init__(self):
        self.requests = {}
        self._lock = asyncio.Lock()  # Async lock for thread safety
    
    async def is_allowed(self, identifier: str, limit: int = SecurityConfig.RATE_LIMIT_REQUESTS) -> bool:
        """Check if request is allowed under rate limit - thread-safe"""
        async with self._lock:
            current_time = time.time()
            window_start = current_time - SecurityConfig.RATE_LIMIT_WINDOW
            
            # Initialize if not exists
            if identifier not in self.requests:
                self.requests[identifier] = []
            
            # Remove old requests outside the window
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier] 
                if req_time > window_start
            ]
            
            # Check if limit exceeded
            if len(self.requests[identifier]) >= limit:
                return False
            
            # Add current request
            self.requests[identifier].append(current_time)
            return True

# Global instances
api_key_auth = APIKeyAuth()
rate_limiter = RateLimiter()

# Dependency functions for FastAPI
async def authenticate(request: Request) -> Dict[str, Any]:
    """FastAPI dependency for authentication - REQUIRED for all endpoints"""
    return await api_key_auth.authenticate_request(request)

async def check_rate_limit(request: Request) -> None:
    """FastAPI dependency for rate limiting - optimized for concurrent requests"""
    # Use API key info if available, otherwise use IP
    identifier = request.client.host if request.client else "unknown"
    limit = SecurityConfig.RATE_LIMIT_REQUESTS
    
    if hasattr(request.state, 'api_key_info'):
        # Use API key specific rate limit
        key_info = request.state.api_key_info
        identifier = key_info['name']
        limit = key_info.get('rate_limit', SecurityConfig.RATE_LIMIT_REQUESTS)
    
    if not await rate_limiter.is_allowed(identifier, limit):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {limit} requests per {SecurityConfig.RATE_LIMIT_WINDOW} seconds",
            headers={"Retry-After": str(SecurityConfig.RATE_LIMIT_WINDOW)}
        )

async def require_permission(permission: str):
    """FastAPI dependency to check specific permissions"""
    async def permission_checker(request: Request):
        if not hasattr(request.state, 'api_key_info'):
            raise HTTPException(status_code=401, detail="Authentication required")
        
        key_info = request.state.api_key_info
        if permission not in key_info.get('permissions', []):
            raise HTTPException(
                status_code=403, 
                detail=f"Permission '{permission}' required"
            )
        return key_info
    
    return permission_checker

# Security utility functions
def generate_api_key(prefix: str = SecurityConfig.API_KEY_PREFIX) -> str:
    """Generate a new API key"""
    return f"{prefix}{secrets.token_urlsafe(32)}"

def hash_api_key(api_key: str) -> str:
    """Hash API key for storage (if storing in database)"""
    return hashlib.sha256(api_key.encode()).hexdigest()

def create_cors_config() -> Dict[str, Any]:
    """Create CORS configuration"""
    return {
        "allow_origins": SecurityConfig.ALLOWED_ORIGINS,
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        "allow_headers": [
            "Accept",
            "Accept-Language", 
            "Content-Language",
            "Content-Type",
            "Authorization",
            SecurityConfig.API_KEY_HEADER,  # Updated to use new header name
            SecurityConfig.SIGNATURE_HEADER,
            SecurityConfig.TIMESTAMP_HEADER,
            "User-Agent",
            "X-Requested-With"
        ],
        "expose_headers": [
            "X-Total-Count",
            "X-Rate-Limit-Remaining",
            "X-Rate-Limit-Reset"
        ]
    }

# Example of how to use request signatures (for your frontend)
def example_frontend_request_with_signature():
    """
    Example of how your frontend should make authenticated requests
    """
    example_code = '''
    // Frontend JavaScript example
    async function makeAuthenticatedRequest(url, data = null) {
        const apiKey = process.env.REACT_APP_VOX_API_KEY; // Your frontend API key
        const secretKey = process.env.REACT_APP_VOX_SECRET_KEY; // Optional: for signatures
        const timestamp = Math.floor(Date.now() / 1000).toString();
        
        const headers = {
            'Content-Type': 'application/json',
            'X-VOX-API-KEY': apiKey,
            'X-VOX-Timestamp': timestamp
        };
        
        // Optional: Add signature for extra security
        if (secretKey && data) {
            const body = JSON.stringify(data);
            const message = timestamp + '.' + body;
            const signature = await crypto.subtle.sign(
                'HMAC',
                await crypto.subtle.importKey(
                    'raw', 
                    new TextEncoder().encode(secretKey),
                    { name: 'HMAC', hash: 'SHA-256' },
                    false,
                    ['sign']
                ),
                new TextEncoder().encode(message)
            );
            const hexSignature = Array.from(new Uint8Array(signature))
                .map(b => b.toString(16).padStart(2, '0')).join('');
            headers['X-VOX-Signature'] = 'sha256=' + hexSignature;
        }
        
        const response = await fetch(url, {
            method: data ? 'POST' : 'GET',
            headers: headers,
            body: data ? JSON.stringify(data) : null
        });
        
        if (!response.ok) {
            throw new Error(`API request failed: ${response.status}`);
        }
        
        return await response.json();
    }
    '''
    return example_code 