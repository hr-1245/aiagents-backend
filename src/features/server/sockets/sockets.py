import json
import socketio
from fastapi import APIRouter, Request
import httpx

router = APIRouter()

GHL_SEND_MESSAGE_ENDPOINT = (
    "https://services.leadconnectorhq.com/conversations/messages"
)

# Create Socket.IO server
sio_server = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=[],
)

# Create ASGI app for Socket.IO
sio_app = socketio.ASGIApp(
    socketio_server=sio_server,
    socketio_path="sockets",
)


# Register events
@sio_server.event
async def connect(sid, environ):
    # token = auth.get("token")
    print(f"🔌 {sid} connected")


@sio_server.event
async def disconnect(sid):
    print(f"❌ {sid} disconnected")


@sio_server.event
async def chat_message(sid, data):
    print("new message data: ", data)

    payload = {
        "contactId": "YneAmPjjLo4ONDkhukEv",
        # "contactId": data["contactId"],
        "message": data["message"],
        "type": data["type"],
    }

    headers = {
        "Authorization": f"Bearer {data['token']}",
        "Content-Type": "application/json",
        "Version": "2021-04-15",
    }

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            GHL_SEND_MESSAGE_ENDPOINT, headers=headers, json=payload
        )
        response.raise_for_status()  # 201 Created expected

    # GHL returns the created message object
    ghl_response = response.json()

    # Send the GHL reply straight back to the same socket
    await sio_server.emit("new_message", ghl_response, room=sid)


@router.post("/webhooks/ghl/message")
async def ghl_webhook(request: Request):
    try:
        body_bytes = await request.body()
        if not body_bytes:
            print("⚠️ Empty webhook request received")
            return {"status": "ok", "message": "Empty request ignored"}

        body = await request.json()
        print("📩 Incoming GHL Webhook:", json.dumps(body, indent=2))

        # Match real payload keys
        contact_id = body.get("contact_id")
        conversation_id = body.get(
            "conversationId"
        )  # only if you add it in Custom Data
        message_text = body.get("message", {}).get("body")
        message_type = body.get("message", {}).get("type")

        standardized_payload = {
            "contactId": contact_id,
            "conversationId": conversation_id,
            "message": message_text,
            "type": message_type,
        }

        await sio_server.emit("new_message", standardized_payload)
        return {"status": "ok"}

    except json.JSONDecodeError:
        print("❌ Invalid JSON received in webhook")
        return {"status": "error", "message": "Invalid JSON format"}
    except Exception as e:
        print("❌ Webhook error:", e)
        return {"status": "error", "message": str(e)}
