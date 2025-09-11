from fastapi import APIRouter, WebSocket
import requests

router = APIRouter()

API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJsb2NhdGlvbl9pZCI6ImlYVG1yQ2tXdFpLWFd6czg1Sng4IiwidmVyc2lvbiI6MSwiaWF0IjoxNzU2NDEzMDkzMjY5LCJzdWIiOiJlbkdqVElOZjNhZWFFSXJxeWRFYiJ9.jA9i_Nwruo2d2mLVhPWRUrApIzwIg2-Orw6Labw2vIw"
BASE_URL = "https://services.leadconnectorhq.com"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Version": "2021-07-28",
    "Content-Type": "application/json",
}

data = {
    "contactId": "YneAmPjjLo4ONDkhukEv",  # real contactId from your GHL account
    "type": "SMS",
    "message": "Hello from FastAPI (dummy test)!",
}


@router.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client connected")

    try:
        while True:
            data = await websocket.receive_text()
            print("data from frontend => ", data)

            # msg = {
            #     "contactId": "YneAmPjjLo4ONDkhukEv",
            #     "type": "SMS",
            #     "message": "Hello from FastAPI!",
            # }

            # url = f"{BASE_URL}/conversations/YneAmPjjLo4ONDkhukEv/messages"
            # print("url => ", url)
            # response = requests.post(url, headers=headers, json=msg)
            # print("response => ", response)

    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        print("🔌 Client disconnected")
