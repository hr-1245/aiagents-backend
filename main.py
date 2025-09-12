
import uvicorn

from src.run_server import app, run_server  # import your FastAPI app + runner
from src.features.server.sockets.sockets import sio_app  # import socket layer

# Mount Socket.IO into your existing FastAPI app
app.mount("/", sio_app)

if __name__ == "__main__":
    # Option 1: use the run_server (with logging, signals, etc.)
    run_server()

    # Option 2 (dev only): directly run with uvicorn
    # uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
