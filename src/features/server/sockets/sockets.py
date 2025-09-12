import socketio

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
async def connect(sid, environ, auth):

    print(f"🔌 {sid} connected")
    print(f"Token: ", auth)


@sio_server.event
async def disconnect(sid):
    print(f"❌ {sid} disconnected")


@sio_server.event
async def chat_message(sid, data):
    print(f"💬 Message from {sid}: {data}")
