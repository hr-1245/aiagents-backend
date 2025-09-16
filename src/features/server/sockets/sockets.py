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


# @sio_server.event
# async def get_all_messages(sid, data):
#     token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdXRoQ2xhc3MiOiJMb2NhdGlvbiIsImF1dGhDbGFzc0lkIjoiaVhUbXJDa1d0WktYV3pzODVKeDgiLCJzb3VyY2UiOiJJTlRFR1JBVElPTiIsInNvdXJjZUlkIjoiNjdmZGYyM2FmZWExMGY0OGZkNzhjZjBiLW05aTJ4YjJ1IiwiY2hhbm5lbCI6Ik9BVVRIIiwicHJpbWFyeUF1dGhDbGFzc0lkIjoiaVhUbXJDa1d0WktYV3pzODVKeDgiLCJvYXV0aE1ldGEiOnsic2NvcGVzIjpbImNvbnZlcnNhdGlvbnMucmVhZG9ubHkiLCJjb252ZXJzYXRpb25zL21lc3NhZ2UucmVhZG9ubHkiLCJjb252ZXJzYXRpb25zL3JlcG9ydHMucmVhZG9ubHkiLCJjb250YWN0cy5yZWFkb25seSIsImxvY2F0aW9ucy5yZWFkb25seSIsImNvbnZlcnNhdGlvbnMvbWVzc2FnZS53cml0ZSIsImNvbnZlcnNhdGlvbnMvbGl2ZWNoYXQud3JpdGUiLCJjb252ZXJzYXRpb25zLndyaXRlIiwib2JqZWN0cy9zY2hlbWEud3JpdGUiXSwiY2xpZW50IjoiNjdmZGYyM2FmZWExMGY0OGZkNzhjZjBiIiwidmVyc2lvbklkIjoiNjdmZGYyM2FmZWExMGY0OGZkNzhjZjBiIiwiY2xpZW50S2V5IjoiNjdmZGYyM2FmZWExMGY0OGZkNzhjZjBiLW05aTJ4YjJ1In0sImlhdCI6MTc1NzkzMDU5Mi4wNiwiZXhwIjoxNzU4MDE2OTkyLjA2fQ.vlbU_38A7yDS0vwYmWUt2mxX5cEZusghu_430VkHAr_Spu5WwALcjHsK2RygLQcmvtoIbpiIHRLvcPNlDyvhQCcmiDXVJ6-ogC1Wf-uFGLagC9PSJyoOriS0lFJd6Tp3ZH4TLR2CTrpV9bsSr8zk0UQMordGO70zCNJOFqGecH85sbEJVANVU84Y42wYlJiBqbFabICF505MVE9sNdnTu-ciuHLYMocuRIHWk9KUYPdW8NNzSsMy1kGBixX7QZI_arWJHMZT-5rcj0F6ifrCDg_t9EzjQ6yDoeFz4W-iuqkI7UppDQIwNed_LPHu6ZVpe2fkpvw0spj5lYfujCGKK1Qn-cstOa8tzVFQ8zQgZUQWYJuIHeNqVHc44gqXwG8CLs5FA7_I3_p4-oz_S6Pu-95-yzdmMUv2-6augrE8gk26E37LvWs9gI3oE2chqKzMgcJ0OjVUTiy_GdvB-g-BcFY_3s5WPqqER97BMBMt3R3fMYum3HH2jIATUkRUx4k0yetDrwTAkyliJQN-ndcgmeNC3225JOWzBZezl9jAma-Eys5XpjKbRWkEoO5mJRGSroXx1ek4ot3vyK9IMYslViL2Qao7rxC41byadv4agug5f5fVui7SteEFJ_z6xWj_SJ0t5BmRM5QFYijjc7axtB48cLHV3m_NIMuUpbYPhCE"
#     conversation_id = "fHtv0rSh2Pde7sewlSdL"
#     # print("data: ", data)

#     headers = {
#         # "Authorization": f"Bearer {data['token']}",
#         "Authorization": f"Bearer {token}",
#         "Content-Type": "application/json",
#         "Version": "2021-04-15",
#     }
#     # print("headers: ", headers)

#     GHL_GET_MESSAGES_ENDPOINT = (
#         f"https://services.leadconnectorhq.com/conversations/{conversation_id}/messages"
#     )
#     # GHL_GET_MESSAGES_ENDPOINT = f"https://services.leadconnectorhq.com/conversations/{data["conversationId"]}/messages"

#     async with httpx.AsyncClient(timeout=10) as client:
#         response = await client.get(GHL_GET_MESSAGES_ENDPOINT, headers=headers)
#         response.raise_for_status()  # 201 Created expected

#     # GHL returns the created message object
#     ghl_response = response.json()
#     print("ghl_response get all messages:------- :", ghl_response)

#     # Send the GHL reply straight back to the same socket
#     await sio_server.emit("get_all_messages", ghl_response, room=sid)


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
        conversation_id = body.get("conversationId")  # only if you add it in Custom Data
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
