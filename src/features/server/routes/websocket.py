# src/features/server/routes/websocket.py
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List
import json
import asyncio
import httpx
from typing import Any, Dict, Optional

GHL_API_BASE = "https://services.leadconnectorhq.com"

router = APIRouter()

# Track active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# @router.websocket("/ws/train")
# async def websocket_endpoint(websocket: WebSocket):
#     await manager.connect(websocket)
#     try:
#         while True:
#             data = await websocket.receive_text()
#             await manager.send_personal_message(f"You said: {data}", websocket)
#             await manager.broadcast(f"📢 Broadcast: {data}")
#     except WebSocketDisconnect:
#         manager.disconnect(websocket)
#         await manager.broadcast("❌ A client disconnected")
# Simulated token storage / refresh functions (replace with your real logic)
async def get_valid_ghl_tokens(user_id: str) -> Dict[str, str]:
    # Return a dict: {"access_token": "...", "refresh_token": "..."}
    return {"access_token": "YOUR_ACCESS_TOKEN", "refresh_token": "YOUR_REFRESH_TOKEN"}

async def refresh_ghl_tokens(user_id: str, refresh_token: str) -> Dict[str, str]:
    # Implement refresh logic here
    # Example: POST to https://services.leadconnectorhq.com/oauth/refresh ...
    return {"access_token": "NEW_ACCESS_TOKEN", "refresh_token": refresh_token}

async def fetch_ghl_api_with_refresh(
    endpoint: str,
    user_id: str,
    method: str = "GET",
    data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Any:
    url = endpoint if endpoint.startswith("http") else f"{GHL_API_BASE}{endpoint}"
    retry_count = 0
    max_retries = 2

    async with httpx.AsyncClient(timeout=30.0) as client:
        while retry_count <= max_retries:
            tokens = await get_valid_ghl_tokens(user_id)
            access_token = tokens.get("access_token")
            refresh_token = tokens.get("refresh_token")

            if not access_token:
                raise Exception("No valid GHL access token available. Re-authenticate.")

            req_headers = {
                "Authorization": f"Bearer {access_token}",
                "Version": "2021-07-28",
                "Content-Type": "application/json",
            }
            if headers:
                req_headers.update(headers)

            try:
                response = await client.request(
                    method=method.upper(),
                    url=url,
                    headers=req_headers,
                    json=data,
                    params=params,
                )

                if response.status_code == 401 and retry_count < max_retries:
                    # Token expired, refresh
                    if not refresh_token:
                        raise Exception("No refresh token available. Re-authenticate.")

                    new_tokens = await refresh_ghl_tokens(user_id, refresh_token)
                    if not new_tokens.get("access_token"):
                        raise Exception("Failed to refresh GHL token.")

                    retry_count += 1
                    continue  # retry with new token

                response.raise_for_status()
                # Parse JSON
                try:
                    return response.json()
                except json.JSONDecodeError:
                    raise Exception(f"Invalid JSON response: {response.text[:100]}...")

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401 and retry_count < max_retries:
                    retry_count += 1
                    continue
                raise e
            except Exception as e:
                raise e

        raise Exception(f"GHL API request failed after {max_retries + 1} attempts")

# Helper to build query strings
def build_query_string(params: Dict[str, Any]) -> str:
    from urllib.parse import urlencode
    processed = {}
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, list):
            processed[k] = ",".join(map(str, v))
        elif isinstance(v, dict):
            processed[k] = json.dumps(v)
        else:
            processed[k] = str(v)
    return f"?{urlencode(processed)}" if processed else ""

@router.websocket("/ws/train")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                data = json.loads(raw_data)
                event = data.get("event")
                payload = data.get("data")

                if event == "accept_text":
                    # Call GHL API
                    response = await fetch_ghl_api_with_refresh(
                        endpoint="/contacts",
                        user_id="1234",
                        method="GET",
                    )
                    await manager.send_personal_message(
                        f"✅ Accepted: {payload}\nGHL Response: {response}", websocket
                    )
                else:
                    await manager.send_personal_message("❌ Unknown event", websocket)

            except json.JSONDecodeError:
                await manager.send_personal_message("❌ Invalid JSON format", websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast("❌ A client disconnected")

