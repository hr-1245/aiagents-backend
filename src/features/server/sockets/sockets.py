import json
import re
import socketio
from fastapi import APIRouter, Request
import httpx
from src.features.ai.agents.custom_agent_service import logger
from openai import AsyncOpenAI
import os
import chromadb
from supabase import create_client, Client
from dotenv import load_dotenv
from collections import defaultdict
import asyncio
import traceback

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

# Buffer messages per contact_id (list of message texts)
_pending_messages: dict = defaultdict(list)

# Store latest phone and latest tag per contact (so we can use them when replying)
_latest_phone: dict = {}
_latest_tag: dict = {}

# Active asyncio.Task per contact for the debounce timer
_active_tasks: dict = {}

# Debounce delay in seconds
DEBOUNCE_SECONDS = 10

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


# ----- Webhook endpoint (GHL) -----
@router.post("/webhooks/ghl/message")
# @router.post("/test-ghl")
async def ghl_webhook(request: Request):
    try:
        # print("\n🚀 [Webhook] GHL message received")

        # --- 1️⃣ Parse the body ---
        body_bytes = await request.body()
        if not body_bytes:
            # print("⚠️ [Webhook] Empty request body — ignoring")
            return {"status": "ok", "message": "Empty request ignored"}

        body = await request.json()
        # print(f"📩 [Webhook] Raw GHL payload:\n{json.dumps(body, indent=2)}")

        # --- 2️⃣ Extract core fields ---
        contact_id = body.get("contact_id")
        phone = body.get("phone")
        message_text = body.get("message", {}).get("body")
        message_type = body.get("message", {}).get("type")
        ghl_tag = body.get("tags")
        location_id = body.get("location", {}).get("id")

        # print(
        #     f"🧩 [Extracted] contact_id={contact_id}, phone={phone}, tag={ghl_tag}, message={message_text}"
        # )

        if not phone or not message_text:
            # print("⚠️ [Webhook] Missing phone or message_text — skipping processing")
            return {"status": "ok", "message": "Missing phone or message_text"}

        standardized_payload = {
            "contactId": contact_id,
            "phone": phone,
            "message": message_text,
            "type": message_type,
        }

        # --- 2️⃣ Emit new incoming message to frontend (unchanged — immediate) ---
        await sio_server.emit("new_message", standardized_payload)
        # print(
        #     f"📤 [Emit] Forwarded incoming message to frontend → {standardized_payload}"
        # )

        # ----- Debounce buffering logic (NEW) -----
        # Append message_text into pending buffer
        _pending_messages[contact_id].append(message_text)
        # Keep latest phone and tag for when we process
        _latest_phone[contact_id] = phone
        _latest_tag[contact_id] = ghl_tag

        # If there's an existing active debounce task for this contact, cancel it
        existing_task = _active_tasks.get(contact_id)

        print(
            "============================= existing_task =============================> ",
            existing_task,
        )
        if existing_task and not existing_task.done():
            try:
                existing_task.cancel()
                print(
                    f"🔁 [Debounce] Cancelled existing debounce task for {contact_id}"
                )
            except Exception as e:
                print(f"⚠️ [Debounce] Error cancelling task for {contact_id}: {e}")

        # Start a new debounce task
        task = asyncio.create_task(_debounced_process(contact_id, location_id))
        _active_tasks[contact_id] = task
        print(
            f"⏱️ [Debounce] Started debounce task for {contact_id} (waiting {DEBOUNCE_SECONDS}s)"
        )

        # Return immediately — processing happens in background task
        return {"status": "ok", "message": "Received and buffered"}

    except Exception as e:
        # print("❌ [ERROR] ghl_webhook:", str(e))

        # print(traceback.format_exc())
        return {"status": "error", "message": str(e)}


# ----- Helper: the AI + send-to-GHL flow (refactored from your original code) -----
async def _process_ai_and_send(
    contact_id: str, phone: str, message_text: str, ghl_tag, location_id: str
):
    """
    This function contains the existing logic you had for:
      - matching agent
      - preparing AI config
      - embeddings, KB query
      - chat completion
      - send reply to GHL
    It is adapted to accept the combined message_text.
    """
    try:
        # print(
        #     f"\n🚀 [_process_ai_and_send] Processing messages for contact {contact_id}"
        # )
        # --- Match agent using tag ---
        if not ghl_tag:
            # print("⚠️ [_process_ai_and_send] No tag provided — cannot match agent")
            return {"status": "ok", "message": "No tag provided"}

        # print(f"🔍 [DB] Searching for AI agent with tag: '{ghl_tag}' ...")

        # Split incoming tags by comma or dot
        tags = [t.strip() for t in re.split(r"[,.]", ghl_tag) if t.strip()]

        response = supabase.table("ai_agents").select("*").execute()
        all_agents = response.data or []

        matching_agents = []
        for tag in tags:
            # print(f"🔍 Checking agents for tag: '{tag}'")

            # Find agents with current tag
            matching_agents = [
                agent for agent in all_agents if agent.get("data", {}).get("tag") == tag
            ]

            if matching_agents:
                # print(
                #     f"🎯 Found {len(matching_agents)} agent(s) for tag '{tag}' — stopping further checks."
                # )
                break  # Stop after finding matches for first valid tag

        # if not matching_agents:
        #     print("⚠️ No matching agents found for any tag.")

        # print(f"📦 [DB] Found total {len(all_agents)} agents in database")

        # matching_agents = [
        #     agent for agent in all_agents if agent.get("data", {}).get("tag") == ghl_tag
        # ]
        # print(f"🎯 [DB] Matched {len(matching_agents)} agent(s) with tag '{ghl_tag}'")

        # if not matching_agents:
        #     print("⚠️ [_process_ai_and_send] No agent found with this tag")
        #     return {"status": "ok", "message": "No matching agent found"}

        agent = matching_agents[0]
        # print(
        #     f"🤖 [Agent] Selected agent → {agent.get('name')} (ID: {agent.get('id')})"
        # )

        # Prepare AI configuration
        agent_name = agent.get("name", "AI Assistant")
        agent_personality = agent.get(
            "system_prompt", "You are a helpful AI assistant."
        )
        agent_intent = agent.get("intent", "Assist the user helpfully.")
        response_config = agent.get("responseConfig", {}) or {}

        model = response_config.get("model", "gpt-4o-mini")
        temperature = response_config.get("temperature", 0.7)
        kb_ids = agent.get("knowledge_base_ids", [])

        # print(f"⚙️ [AI Config] model={model}, temperature={temperature}, KBs={kb_ids}")

        # Create embedding for message
        # print("🧬 [Embedding] Generating embedding for incoming message...")
        embedding = await client.embeddings.create(
            input=message_text,
            model="text-embedding-3-small",
            encoding_format="float",
        )
        query_vector = embedding.data[0].embedding
        # print("✅ [Embedding] Embedding generated successfully")

        # Query Knowledge Base
        retrieved_docs = []
        if kb_ids:
            # print(f"📚 [KB] Querying {len(kb_ids)} knowledge base(s)...")
            for kb_id in kb_ids:
                try:
                    collection = chroma_client.get_or_create_collection(name=kb_id)
                    # print(f"➡️ [KB] Checking collection '{kb_id}' ...")
                    results = collection.query(
                        query_embeddings=[query_vector], n_results=5
                    )

                    if results and results.get("documents"):
                        docs = results["documents"][0]
                        retrieved_docs.extend(docs)
                        # print(
                        #     f"✅ [KB] Retrieved {len(docs)} documents from KB {kb_id}"
                        # )
                    else:
                        print(f"⚠️ [KB] No documents found in KB {kb_id}")
                except Exception as chroma_err:
                    print(f"❌ [KB] Error querying KB {kb_id}: {str(chroma_err)}")
        else:
            print("ℹ️ [KB] No KBs found for this agent — fallback mode")

        # Generate AI Reply
        # print("🧠 [AI] Generating AI response...")
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
            # print("💬 [AI] Generated fallback response successfully")
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
            # print("💬 [AI] Generated contextual response successfully")

        # Emit auto-reply (send to GHL)
        reply_payload = {
            "contactId": contact_id,
            "phone": phone,
            "message": reply,
            "type": "WhatsApp",
        }

        token_response = (
            supabase.table("provider_data")
            .select("token")
            .eq("location_id", location_id)
            .single()
            .execute()
        )
        provider = token_response.data
        access_token = provider.get("token") if provider else None

        # print("access_token ===================================> ", access_token)

        headers = {
            # Keep your existing headers — tokens, etc.
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Version": "2021-04-15",
        }

        async with httpx.AsyncClient(timeout=10) as http_client:
            response = await http_client.post(
                GHL_SEND_MESSAGE_ENDPOINT, headers=headers, json=reply_payload
            )
            response.raise_for_status()
            # print(f"📤 [GHL] Sent auto-reply message to GHL for contact {contact_id}")

        # print(f"🚀 [Emit] Sending auto reply via socket: {reply_payload}")
        await sio_server.emit("new_message", reply_payload)

        # print("✅ [_process_ai_and_send] Auto reply emitted successfully")
        # print("===============================================================")

        return {"status": "ok", "autoReply": reply}

    except Exception as e:
        # print("❌ [_process_ai_and_send] Error:", str(e))

        # print(traceback.format_exc())
        return {"status": "error", "message": str(e)}

    # ----- Debounce worker -----


async def _debounced_process(contact_id: str, location_id: str):
    """
    Waits DEBOUNCE_SECONDS since last schedule, then processes all buffered
    messages for the contact as a single combined message.
    """
    try:
        # Wait; this task may be cancelled if a new message arrives
        await asyncio.sleep(DEBOUNCE_SECONDS)
    except asyncio.CancelledError:
        # Task was cancelled because a new message arrived — nothing to do
        # print(f"🛑 [_debounced_process] Cancelled debounce task for {contact_id}")
        return

    try:
        # Pop the buffer and metadata for this contact
        messages = _pending_messages.pop(contact_id, [])
        phone = _latest_phone.pop(contact_id, None)
        ghl_tag = _latest_tag.pop(contact_id, None)

        # clear active task reference
        _active_tasks.pop(contact_id, None)

        if not messages:
            print(f"⚠️ [_debounced_process] No messages to process for {contact_id}")
            return

        # Decide how to combine messages: join with newline (you can change)
        combined_text = "\n".join(messages)
        # print(f"🕓 [_debounced_process] Debounce window ended for {contact_id}.")
        # print(f"📥 [_debounced_process] Messages combined:\n{combined_text}")

        # Call the original AI + send flow
        await _process_ai_and_send(
            contact_id, phone, combined_text, ghl_tag, location_id
        )

    except Exception as e:
        # print(f"❌ [_debounced_process] Error while processing {contact_id}: {e}")

        print(traceback.format_exc())
