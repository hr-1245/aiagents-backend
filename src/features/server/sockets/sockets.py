import json
import socketio
from fastapi import APIRouter, Request
import httpx
from src.features.ai.agents.custom_agent_service import logger
from openai import AsyncOpenAI
import os
import chromadb
from supabase import create_client, Client
from dotenv import load_dotenv

router = APIRouter()

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Supabase credentials not configured properly")
    supabase: Client = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Supabase connected successfully")


PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_data")
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

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
        "phone": data["phone"],
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

    # 4️⃣ Inject phone number for frontend consistency
    ghl_get_response["phone"] = payload["phone"]

    print("ghl_get_response ======> ", ghl_get_response)

    # 4. Emit the fetched message
    await sio_server.emit("new_message", ghl_get_response, room=sid)


# @router.post("/test-ghl")
@router.post("/webhooks/ghl/message")
async def ghl_webhook(request: Request):
    try:
        print("\n🚀 [Webhook] GHL message received")

        # --- 1️⃣ Parse the body ---
        body_bytes = await request.body()
        if not body_bytes:
            print("⚠️ [Webhook] Empty request body — ignoring")
            return {"status": "ok", "message": "Empty request ignored"}

        body = await request.json()
        print(f"📩 [Webhook] Raw GHL payload:\n{json.dumps(body, indent=2)}")

        # --- 2️⃣ Extract core fields ---
        contact_id = body.get("contact_id")
        phone = body.get("phone")
        message_text = body.get("message", {}).get("body")
        message_type = body.get("message", {}).get("type")
        ghl_tag = body.get("tags")

        print(
            f"🧩 [Extracted] contact_id={contact_id}, phone={phone}, tag={ghl_tag}, message={message_text}"
        )

        if not phone or not message_text:
            print("⚠️ [Webhook] Missing phone or message_text — skipping processing")
            return {"status": "ok", "message": "Missing phone or message_text"}

        standardized_payload = {
            "contactId": contact_id,
            "phone": phone,
            "message": message_text,
            "type": message_type,
        }

        # --- 2️⃣ Emit new incoming message to frontend ---
        await sio_server.emit("new_message", standardized_payload)
        print(
            f"📤 [Emit] Forwarded incoming message to frontend → {standardized_payload}"
        )

        # --- 3️⃣ Match agent using tag ---
        if not ghl_tag:
            print("⚠️ [Webhook] No tag provided — cannot match agent")
            return {"status": "ok", "message": "No tag provided"}

        print(f"🔍 [DB] Searching for AI agent with tag: '{ghl_tag}' ...")
        response = supabase.table("ai_agents").select("*").execute()
        all_agents = response.data or []
        print(f"📦 [DB] Found total {len(all_agents)} agents in database")

        # Filter based on tag
        matching_agents = [
            agent for agent in all_agents if agent.get("data", {}).get("tag") == ghl_tag
        ]
        print(f"🎯 [DB] Matched {len(matching_agents)} agent(s) with tag '{ghl_tag}'")

        if not matching_agents:
            print("⚠️ [Webhook] No agent found with this tag")
            return {"status": "ok", "message": "No matching agent found"}

        agent = matching_agents[0]
        print(
            f"🤖 [Agent] Selected agent → {agent.get('name')} (ID: {agent.get('id')})"
        )

        # --- 4️⃣ Prepare AI configuration ---
        agent_name = agent.get("name", "AI Assistant")
        agent_personality = agent.get(
            "system_prompt", "You are a helpful AI assistant."
        )
        agent_intent = agent.get("intent", "Assist the user helpfully.")
        response_config = agent.get("responseConfig", {}) or {}

        model = response_config.get("model", "gpt-4o-mini")
        temperature = response_config.get("temperature", 0.7)
        kb_ids = agent.get("knowledge_base_ids", [])

        print(f"⚙️ [AI Config] model={model}, temperature={temperature}, KBs={kb_ids}")

        # --- 5️⃣ Create embedding for message ---
        print("🧬 [Embedding] Generating embedding for incoming message...")
        embedding = await client.embeddings.create(
            input=message_text,
            model="text-embedding-3-small",
            encoding_format="float",
        )
        query_vector = embedding.data[0].embedding
        print("✅ [Embedding] Embedding generated successfully")

        # --- 6️⃣ Query Knowledge Base ---
        retrieved_docs = []
        if kb_ids:
            print(f"📚 [KB] Querying {len(kb_ids)} knowledge base(s)...")
            for kb_id in kb_ids:
                try:
                    collection = chroma_client.get_or_create_collection(name=kb_id)
                    print(f"➡️ [KB] Checking collection '{kb_id}' ...")
                    results = collection.query(
                        query_embeddings=[query_vector], n_results=5
                    )

                    if results and results.get("documents"):
                        docs = results["documents"][0]
                        retrieved_docs.extend(docs)
                        print(
                            f"✅ [KB] Retrieved {len(docs)} documents from KB {kb_id}"
                        )
                    else:
                        print(f"⚠️ [KB] No documents found in KB {kb_id}")
                except Exception as chroma_err:
                    print(f"❌ [KB] Error querying KB {kb_id}: {str(chroma_err)}")
        else:
            print("ℹ️ [KB] No KBs found for this agent — fallback mode")

        # --- 7️⃣ Generate AI Reply ---
        print("🧠 [AI] Generating AI response...")
        if not retrieved_docs:
            fallback_prompt = f"""
            You are {agent_name}.
            Personality: {agent_personality}
            Intent: {agent_intent}

            User said: {message_text}

            Respond naturally, warmly, and helpfully.
            """
            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": agent_personality},
                    {"role": "user", "content": fallback_prompt},
                ],
                temperature=temperature,
            )
            reply = completion.choices[0].message.content
            print("💬 [AI] Generated fallback response successfully")
        else:
            context = "\n\n".join(retrieved_docs[:10])
            context_prompt = f"""
            You are {agent_name}.
            Personality: {agent_personality}
            Intent: {agent_intent}

            Use the following KB context to reply precisely and factually.

            Context:
            {context}

            User query:
            {message_text}
            """
            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": agent_personality},
                    {"role": "user", "content": context_prompt},
                ],
                temperature=temperature,
            )
            reply = completion.choices[0].message.content
            print("💬 [AI] Generated contextual response successfully")

        # --- 8️⃣ Emit auto-reply ---
        reply_payload = {
            "contactId": contact_id,
            "phone": phone,
            "message": reply,
            "type": "WhatsApp",
        }

        headers = {
            "Authorization": f"Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdXRoQ2xhc3MiOiJMb2NhdGlvbiIsImF1dGhDbGFzc0lkIjoiaVhUbXJDa1d0WktYV3pzODVKeDgiLCJzb3VyY2UiOiJJTlRFR1JBVElPTiIsInNvdXJjZUlkIjoiNjdmZGYyM2FmZWExMGY0OGZkNzhjZjBiLW05aTJ4YjJ1IiwiY2hhbm5lbCI6Ik9BVVRIIiwicHJpbWFyeUF1dGhDbGFzc0lkIjoiaVhUbXJDa1d0WktYV3pzODVKeDgiLCJvYXV0aE1ldGEiOnsic2NvcGVzIjpbImNvbnZlcnNhdGlvbnMucmVhZG9ubHkiLCJjb252ZXJzYXRpb25zL21lc3NhZ2UucmVhZG9ubHkiLCJjb252ZXJzYXRpb25zL3JlcG9ydHMucmVhZG9ubHkiLCJjb250YWN0cy5yZWFkb25seSIsImxvY2F0aW9ucy5yZWFkb25seSIsImxvY2F0aW9ucy90YWdzLnJlYWRvbmx5IiwibG9jYXRpb25zL3RhZ3Mud3JpdGUiLCJjb252ZXJzYXRpb25zL21lc3NhZ2Uud3JpdGUiLCJjb252ZXJzYXRpb25zL2xpdmVjaGF0LndyaXRlIiwiY29udmVyc2F0aW9ucy53cml0ZSIsIm9iamVjdHMvc2NoZW1hLndyaXRlIl0sImNsaWVudCI6IjY3ZmRmMjNhZmVhMTBmNDhmZDc4Y2YwYiIsInZlcnNpb25JZCI6IjY3ZmRmMjNhZmVhMTBmNDhmZDc4Y2YwYiIsImNsaWVudEtleSI6IjY3ZmRmMjNhZmVhMTBmNDhmZDc4Y2YwYi1tOWkyeGIydSJ9LCJpYXQiOjE3NjA5MTQ2MTUuMTgxLCJleHAiOjE3NjEwMDEwMTUuMTgxfQ.RiOA78Q661dZvWWaHEuhNWNfeFyntLu7QabBNsLXIkxfirbcGy70-K8uK50C_c6x7hd5irq96hkXRCp0ATRG4REVRo9nNb0xFHdpwf6MBzqK1nWAC9xs8KKyYpbURHEQ4uOu18PpUMKjVGep3TlajaeTMHOJ1M354TL3-PJkepnutTqKyHYMyMlRFD8jW2O_C_MbIMitI_YJlxYg6sbVaSwmaqmYrTT2MXFS3VTasfDEqBosKUKfQViu64Cs8cYCDIA02ntJ2Ys5_BbDUutIomeocxL18qGYRjUGWJRV1TplTQjCZtRfnxy5WRcDdK9rEhPJ_wzJfSJoybErQvNvDkIko5aMo5LgjxyEb89gGPt7XCOtn9C_tW33bkyJaxGuFYIVr_7C6xzs3nCVj2RzY3vePIzpbZtzj9ItWH_yPE8isteHf4sewp6mGoe-DVuxBbNipgU8G8Gk0ujsO1ksE_Tc5mm-Uc0KZKtzv4Zui1ENa3JQ7mXqAdNFsb4HaIgdI0Rj2MlW4ulQxk9Ytjti9Yv3UH5T5FdLlAxzDM47CIRIanadX3OUYg6uLcqaBPuyIO029dpjqoKrZTXFTezGn7zHQ4moZ4ptK3eDp1wCSupQE3R7BRhMGBae_zrX-W2rPt_TY2z73OD_UQPzdLf6j5i3mAkq3-eETzYofh7UdaI",
            "Content-Type": "application/json",
            "Version": "2021-04-15",
        }

        async with httpx.AsyncClient(timeout=10) as http_client:
            response = await http_client.post(
                GHL_SEND_MESSAGE_ENDPOINT, headers=headers, json=reply_payload
            )
            response.raise_for_status()
            print(f"📤 [GHL] Sent auto-reply message to GHL for contact {contact_id}")

        print(f"🚀 [Emit] Sending auto reply via socket: {reply_payload}")
        await sio_server.emit("new_message", reply_payload)

        print("✅ [Webhook] Auto reply emitted successfully")
        print("===============================================================")

        return {"status": "ok", "autoReply": reply}

    except Exception as e:
        print("❌ [ERROR] ghl_webhook:", str(e))
        import traceback

        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}
