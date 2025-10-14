import json
import socketio
from fastapi import APIRouter, Request
import httpx
from src.features.ai.agents.custom_agent_service import logger

router = APIRouter()

GHL_SEND_MESSAGE_ENDPOINT = (
    "https://services.leadconnectorhq.com/conversations/messages"
)
GHL_GET_MESSAGE_ENDPOINT = "https://services.leadconnectorhq.com/conversations/messages"

# Create Socket.IO server
sio_server = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    # cors_allowed_origins=[],
)

# Create ASGI app for Socket.IO
sio_app = socketio.ASGIApp(
    socketio_server=sio_server,
    socketio_path="sockets",
)

# ✅ Store userId → socketId mapping
connected_clients = {}


@sio_server.event
async def connect(sid, environ):
    query = environ.get("QUERY_STRING", "")
    params = dict(pair.split("=") for pair in query.split("&") if "=" in pair)
    user_id = params.get("userId")

    if user_id:
        connected_clients[user_id] = sid
        logger.info(f"✅ Auto-registered user {user_id} on connect with socket {sid}")
    else:
        print(f"⚠️ No userId in connect query for socket {sid}")


@sio_server.event
async def disconnect(sid):
    for user_id, s in list(connected_clients.items()):
        if s == sid:
            del connected_clients[user_id]
            print(f"❌ {user_id} disconnected — removed from connected_clients")
            break


@sio_server.event
async def chat_message(sid, data):

    payload = {
        "contactId": data["contactId"],
        "message": data["message"],
        "type": data["type"],
    }

    headers = {
        "Authorization": f"Bearer {data['token']}",
        "Content-Type": "application/json",
        "Version": "2021-04-15",
    }

    async with httpx.AsyncClient(timeout=10) as client:
        # 1. Send message
        response = await client.post(
            GHL_SEND_MESSAGE_ENDPOINT, headers=headers, json=payload
        )
        response.raise_for_status()
        ghl_response = response.json()

    # 2. Extract messageId
    message_id = ghl_response.get("id") or ghl_response.get("messageId")
    if not message_id:
        print("⚠️ No messageId found in GHL response:", ghl_response)
        await sio_server.emit("new_message", ghl_response, room=sid)
        return

    # 3. Fetch single message (new async client context!)
    async with httpx.AsyncClient(timeout=10) as client:
        url = f"{GHL_GET_MESSAGE_ENDPOINT}/{message_id}"

        response = await client.get(url, headers=headers)
        response.raise_for_status()
        ghl_get_response = response.json()

    print("ghl_get_response ======> ", ghl_get_response)

    # 4. Emit the fetched message
    await sio_server.emit("new_message", ghl_get_response, room=sid)


@router.post("/webhooks/ghl/message")
async def ghl_webhook(request: Request):
    try:
        body_bytes = await request.body()
        if not body_bytes:
            print("⚠️ Empty webhook request received")
            return {"status": "ok", "message": "Empty request ignored"}

        body = await request.json()
        print("📩 Incoming GHL Webhook:", json.dumps(body, indent=2))

        # Match real payload keys from Customer Replied
        contact_id = body.get("contact", {}).get("id")
        conversation_id = body.get("message", {}).get("conversation_id")
        message_text = body.get("message", {}).get("body")
        message_type = body.get("message", {}).get("type")

        standardized_payload = {
            "contactId": contact_id,
            "conversationId": conversation_id,
            "message": message_text,
            "type": message_type,
        }

        print("✅ Emitting new_message:", standardized_payload)
        await sio_server.emit("new_message", standardized_payload)
        print("✅ Emitted successfully")

        return {"status": "ok"}

    except json.JSONDecodeError:
        print("❌ Invalid JSON received in webhook")
        return {"status": "error", "message": "Invalid JSON format"}
    except Exception as e:
        print("❌ Webhook error:", e)
        return {"status": "error", "message": str(e)}
