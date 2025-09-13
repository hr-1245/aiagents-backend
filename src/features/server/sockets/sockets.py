import socketio
import httpx

GHL_ENDPOINT = "https://services.leadconnectorhq.com/conversations/messages"
GHL_ACCESS_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdXRoQ2xhc3MiOiJMb2NhdGlvbiIsImF1dGhDbGFzc0lkIjoiaVhUbXJDa1d0WktYV3pzODVKeDgiLCJzb3VyY2UiOiJJTlRFR1JBVElPTiIsInNvdXJjZUlkIjoiNjdmZGYyM2FmZWExMGY0OGZkNzhjZjBiLW05aTJ4YjJ1IiwiY2hhbm5lbCI6Ik9BVVRIIiwicHJpbWFyeUF1dGhDbGFzc0lkIjoiaVhUbXJDa1d0WktYV3pzODVKeDgiLCJvYXV0aE1ldGEiOnsic2NvcGVzIjpbImNvbnZlcnNhdGlvbnMucmVhZG9ubHkiLCJjb252ZXJzYXRpb25zL21lc3NhZ2UucmVhZG9ubHkiLCJjb252ZXJzYXRpb25zL3JlcG9ydHMucmVhZG9ubHkiLCJjb250YWN0cy5yZWFkb25seSIsImxvY2F0aW9ucy5yZWFkb25seSIsImNvbnZlcnNhdGlvbnMvbWVzc2FnZS53cml0ZSIsImNvbnZlcnNhdGlvbnMvbGl2ZWNoYXQud3JpdGUiLCJjb252ZXJzYXRpb25zLndyaXRlIiwib2JqZWN0cy9zY2hlbWEud3JpdGUiXSwiY2xpZW50IjoiNjdmZGYyM2FmZWExMGY0OGZkNzhjZjBiIiwidmVyc2lvbklkIjoiNjdmZGYyM2FmZWExMGY0OGZkNzhjZjBiIiwiY2xpZW50S2V5IjoiNjdmZGYyM2FmZWExMGY0OGZkNzhjZjBiLW05aTJ4YjJ1In0sImlhdCI6MTc1Nzc3MTMyNi4wNjgsImV4cCI6MTc1Nzg1NzcyNi4wNjh9.f3sPxxmyxURY98x9k2qFcRZQMDw_yvgZ-TnZNkULAXJehRgc4PbiXgvKQQXSXLd3JWa5h6locnbn0inFkJGBUceao8kOsR2vhFBRnFjOewhtTOU1o8gvO_Errj1fqOCk_iKTVUJxeoGTBgralrliP5Z5WS4BtYlAQaCdEtINhZyEmqdX6uS9VwbfG-zlgHOnb_5cHFYxx4n1KxJ2JL4EPfRFIZRhT8tLLXFUnkB8y_ilJF2XvhXD4Yh_udgFIBzN3GkDbAYgdbdREf5fQfXgaM80hfnLCdoPI2K9DpOQ_3x5E2EiZI1y3sEa7Xl-5RoH9UBYRLGRBo7R8UdeFbGBbqXK69r1ZVZIQTpMM2wCUDuWcG7Si38eO9z8C4-DA56eRdKgZSfVGwq8YIE_mNuTIfhy5DxBBBgoO-SNHkNWmnemZPtWtoTbohq7LQ4KEfTQgPsWheSq7zlx_xEEaAl59SMMo7QY0JT0De_kY5iIF85IulRz-1XRcP8RMABTJRCUQCgDS4Vta_9W3rbe50QnuUnh1FmMzW5UhvceBi9xcu16A4mpLE3mM0cErXgbieVfgtMyCJZONDkE1MgUL0wfsDUwCZ-UIUaclmQmg8OEYbIPsnOagZteqHbIzrIUTOVvaomWhTvavFCOLGI3fplcfV1kgxO70OYM7OpL5Cw85O8"

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
async def connect(sid, environ, auh):
    # token = auth.get("token")
    print(f"🔌 {sid} connected")


@sio_server.event
async def disconnect(sid):
    print(f"❌ {sid} disconnected")


@sio_server.event
async def chat_message(sid, data):
    print("message data => ", data)

    payload = {
        "contactId": "YneAmPjjLo4ONDkhukEv",  # or map to real GHL contact id
        "message": data["message"],
        "type": data["type"],
    }
    print("payload: ", payload)

    headers = {
        "Authorization": f"Bearer {GHL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
        "Version": "2021-04-15",
    }
    print("headers: ", headers)

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(GHL_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # 201 Created expected

    # GHL returns the created message object
    ghl_response = response.json()
    print("ghl_response:", ghl_response)

    # Send the GHL reply straight back to the same socket
    await sio_server.emit("ghl_response", ghl_response, room=sid)
