from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, HttpUrl
from src.features.server.sockets.sockets import sio_server, connected_clients
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI
import chromadb
import os
from src.features.ai.agents.custom_agent_service import logger
import aiohttp, bs4, re
from pypdf import PdfReader

router = APIRouter()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_data")
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)


class FileTrainingRequest(BaseModel):
    userId: str
    knowledgebaseId: str
    fileId: str
    fileName: str
    fileType: str
    fileSize: int
    storagePath: str
    supabaseBucket: str
    fileUrl: str


class WebCrawlRequest(BaseModel):
    userId: str
    knowledgebaseId: str
    url: HttpUrl


class FaqTrainingRequest(BaseModel):
    userId: str
    knowledgebaseId: str
    faqs: list[dict]


class QueryRequest(BaseModel):
    input: str
    model: str = "text-embedding-ada-002"
    kbIds: list[str]


async def emit_progress(user_id: str, event: str, payload: dict):
    """Helper to safely emit to a user's socket if connected."""
    sid = connected_clients.get(user_id)
    if sid:
        await sio_server.emit(event, payload, room=sid)
    else:
        print(f"⚠️ No active socket for user {user_id}")


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file using pypdf.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Cleaned extracted text.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    extracted_text = []

    try:
        reader = PdfReader(file_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                extracted_text.append(text.strip())
            else:
                print(f"⚠️ Warning: No extractable text on page {page_num}")

        final_text = "\n".join(extracted_text).strip()
        if not final_text:
            raise ValueError("No extractable text found in the entire PDF.")

        return final_text

    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")


@router.post("/kb/add-files")
async def train_supabase_file(
    body: FileTrainingRequest, authorization: str = Header(...)
):
    try:
        # Emit start
        await emit_progress(
            body.userId,
            "file_status",
            {
                "status": "starting",
                "message": f"Processing {body.fileName}...",
                "fileName": body.fileName,
            },
        )

        # Download file temporarily
        temp_path = f"/tmp/{body.fileName}"
        async with aiohttp.ClientSession() as session:
            async with session.get(body.fileUrl) as resp:
                if resp.status != 200:
                    raise HTTPException(
                        status_code=400, detail="Failed to download file."
                    )
                with open(temp_path, "wb") as f:
                    f.write(await resp.read())

        # Extract text
        text = extract_text_from_pdf(temp_path)

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)

        await emit_progress(
            body.userId,
            "file_status",
            {
                "status": "chunking",
                "fileName": body.fileName,
                "message": f"Split into {len(chunks)} chunks.",
            },
        )

        # 2️⃣ Embedding
        collection = chroma_client.get_or_create_collection(name=body.knowledgebaseId)

        for idx, chunk in enumerate(chunks):
            emb_resp = await client.embeddings.create(
                input=chunk, model="text-embedding-ada-002"
            )
            emb = emb_resp.data[0].embedding

            collection.add(
                documents=[chunk],
                embeddings=[emb],
                metadatas=[{"source": body.fileName, "chunk": idx}],
                ids=[f"{body.fileId}_{idx}"],
            )

            if idx % 5 == 0:  # update every few chunks
                await emit_progress(
                    body.userId,
                    "file_status",
                    {
                        "status": "embedding",
                        "fileName": body.fileName,
                        "message": f"Embedding chunk {idx + 1}/{len(chunks)}...",
                    },
                )

        # 3️⃣ Completed
        await emit_progress(
            body.userId,
            "file_status",
            {
                "status": "completed",
                "message": f"✅ {body.fileName} successfully processed!",
                "fileName": body.fileName,
            },
        )

        return {"success": True}

    except Exception as e:
        await emit_progress(
            body.userId,
            "file_status",
            {
                "status": "error",
                "message": str(e),
                # "fileName": body.fileName,
            },
        )
        raise HTTPException(status_code=500, detail=f"File processing failed: {e}")


@router.post("/kb/add-web")
async def add_web_source(body: WebCrawlRequest, authorization: str = Header(...)):
    """Scrape 1 page, chunk, embed, store, emit — all in-place."""
    try:
        # 0. start
        await emit_progress(
            body.userId,
            "link_status",
            {
                "status": "starting",
                "url": str(body.url),
                "message": f"Scraping {body.url}...",
            },
        )

        # 1. fetch raw HTML
        async with aiohttp.ClientSession() as session:
            async with session.get(str(body.url), timeout=10) as resp:
                resp.raise_for_status()
                html = await resp.text()

        # 2. strip scripts / styles
        soup = bs4.BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)

        if not text or len(text) < 50:
            raise ValueError("Page too small or empty")

        # 3. chunk (same splitter you use for files)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)

        await emit_progress(
            body.userId,
            "link_status",
            {
                "status": "chunking",
                "url": str(body.url),
                "message": f"Split into {len(chunks)} chunks",
            },
        )

        # 4. embed + store (same Chroma collection as files)
        collection = chroma_client.get_or_create_collection(name=body.knowledgebaseId)

        await emit_progress(
            body.userId,
            "link_status",
            {
                "status": "embedding",
                "url": str(body.url),
                "message": f"Embedding {len(chunks)} chunks...",
            },
        )

        for idx, chunk in enumerate(chunks):
            emb_resp = await client.embeddings.create(
                input=chunk, model="text-embedding-ada-002"
            )
            emb = emb_resp.data[0].embedding
            collection.add(
                documents=[chunk],
                embeddings=[emb],
                metadatas=[{"source": str(body.url), "chunk": idx}],
                ids=[f"web_{body.knowledgebaseId}_{idx}"],
            )

        # 5. done
        await emit_progress(
            body.userId,
            "link_status",
            {
                "status": "completed",
                "url": str(body.url),
                "message": f"✅ {body.url} crawled & embedded!",
            },
        )
        return {"success": True}

    except Exception as e:
        await emit_progress(
            body.userId,
            "link_status",
            {
                "status": "error",
                "url": str(body.url),
                "message": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=f"Crawl failed: {e}")


@router.post("/kb/add-faqs")
async def train_faqs(body: FaqTrainingRequest, authorization: str = Header(...)):
    """Chunk, embed, store FAQs exactly like files & web."""

    try:
        url = f"faq:{body.knowledgebaseId}"  # dummy filename for badge
        await emit_progress(
            body.userId,
            "faq_status",
            {
                "status": "starting",
                "fileName": url,
                "message": f"Processing {len(body.faqs)} FAQs...",
            },
        )

        # 1. build one doc per FAQ (Q + A)
        docs = [f"Q: {faq['question']}\nA: {faq['answer']}" for faq in body.faqs]

        # 2. chunk
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = []
        for doc in docs:
            chunks.extend(splitter.split_text(doc))

        await emit_progress(
            body.userId,
            "faq_status",
            {
                "status": "chunking",
                "fileName": url,
                "message": f"Split into {len(chunks)} chunks",
            },
        )

        # 3. embed + store (same Chroma collection)
        collection = chroma_client.get_or_create_collection(name=body.knowledgebaseId)

        await emit_progress(
            body.userId,
            "faq_status",
            {
                "status": "embedding",
                "fileName": url,
                "message": f"Embedding {len(chunks)} chunks...",
            },
        )

        for idx, chunk in enumerate(chunks):
            emb_resp = await client.embeddings.create(
                input=chunk, model="text-embedding-ada-002"
            )
            emb = emb_resp.data[0].embedding
            collection.add(
                documents=[chunk],
                embeddings=[emb],
                metadatas=[{"source": "faq", "chunk": idx}],
                ids=[f"faq_{body.knowledgebaseId}_{idx}"],
            )

        # 4. done
        await emit_progress(
            body.userId,
            "faq_status",
            {
                "status": "completed",
                "fileName": url,
                "message": f"✅ {len(body.faqs)} FAQs embedded!",
            },
        )
        return {"success": True}

    except Exception as e:
        await emit_progress(
            body.userId,
            "faq_status",
            {
                "status": "error",
                "fileName": f"faq:{body.knowledgebaseId}",
                "message": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=f"FAQ processing failed: {e}")


# @router.post("/kb/query")
# async def query_kb(req: QueryRequest):
#     try:
#         # 🧠 0️⃣ Handle case: no KBs provided
#         if not req.kbIds:
#             logger.info("⚠️ No KBs provided. Switching to general AI mode.")
#             return await handle_general_ai_mode(req.input)

#         # 1️⃣ Create embedding for the query
#         embedding = await client.embeddings.create(
#             input=req.input,
#             model=req.model,
#             encoding_format="float",
#         )
#         query_vector = embedding.data[0].embedding

#         # 2️⃣ Retrieve top matches from KBs
#         retrieved_docs = []
#         for kb_id in req.kbIds:
#             try:
#                 collection = chroma_client.get_or_create_collection(name=kb_id)
#                 results = collection.query(query_embeddings=[query_vector], n_results=5)

#                 if results and results.get("documents") and results["documents"][0]:
#                     docs = results["documents"][0]
#                     retrieved_docs.extend(docs)
#                     logger.info(f"✅ Retrieved {len(docs)} docs from KB: {kb_id}")
#                 else:
#                     logger.info(f"⚠️ No documents found in KB: {kb_id}")

#             except Exception as chroma_err:
#                 logger.error(f"❌ Error querying KB {kb_id}: {str(chroma_err)}")

#         # 3️⃣ If no relevant KB data found, use fallback AI mode
#         if not retrieved_docs:
#             logger.info("⚠️ No relevant KB data found. Switching to general AI mode.")
#             return await handle_general_ai_mode(req.input)

#         # 4️⃣ Build context safely
#         context = "\n\n".join(retrieved_docs[:10])

#         # 5️⃣ Create the prompt
#         prompt = f"""
#         You are an AI assistant that answers questions based on the provided context.
#         Use the information below to give a helpful and clear answer.
#         If the context doesn't directly contain the answer, you can still try to infer or summarize helpfully
#         — but avoid making up facts.

#         Context:
#         {context}

#         Question:
#         {req.input}

#         Answer clearly and helpfully:
#         """

#         # 6️⃣ Generate AI response using KB
#         completion = await client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "You are a knowledgeable assistant that answers using provided KB data.",
#                 },
#                 {"role": "user", "content": prompt},
#             ],
#             temperature=0.3,
#         )

#         reply = completion.choices[0].message.content.strip()
#         logger.info(f"✅ AI response generated from KB context.")

#         return {"reply": reply, "mode": "kb"}

#     except Exception as e:
#         logger.exception("❌ QUERY ERROR")
#         raise HTTPException(status_code=500, detail=str(e))


# # -----------------------------
# # 🧩 HELPER: Fallback general mode
# # -----------------------------
# async def handle_general_ai_mode(user_input: str):
#     """Handles case when no KB data is available."""
#     try:
#         fallback_prompt = f"""
#         You are an AI assistant helping users even when no company-specific knowledge base is available.
#         Use your general reasoning to answer clearly, accurately, and helpfully.
#         If the question is unclear or needs company-specific data, politely mention that.

#         Question:
#         {user_input}
#         """

#         completion = await client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "You are a helpful and knowledgeable assistant.",
#                 },
#                 {"role": "user", "content": fallback_prompt},
#             ],
#             temperature=0.5,
#         )

#         reply = completion.choices[0].message.content.strip()
#         logger.info("✅ General AI mode reply generated.")
#         return {"reply": reply, "mode": "general"}

#     except Exception as e:
#         logger.error(f"❌ Error in general AI mode: {str(e)}")
#         raise HTTPException(status_code=500, detail="General AI mode failed.")


@router.post("/kb/query")
async def query_kb(req: QueryRequest):
    try:
        # 1️⃣ Create embedding for the query
        embedding = await client.embeddings.create(
            input=req.input,
            model=req.model,
            encoding_format="float",
        )
        query_vector = embedding.data[0].embedding

        retrieved_docs = []

        # 2️⃣ Handle optional KBs
        if req.kbIds and len(req.kbIds) > 0:
            for kb_id in req.kbIds:
                try:
                    collection = chroma_client.get_or_create_collection(name=kb_id)
                    results = collection.query(
                        query_embeddings=[query_vector],
                        n_results=5,
                    )

                    if results and results.get("documents"):
                        docs = results["documents"][0]
                        retrieved_docs.extend(docs)
                    else:
                        logger.info(f"⚠️ No documents found in KB: {kb_id}")

                except Exception as chroma_err:
                    logger.error(f"❌ Error querying KB {kb_id}: {str(chroma_err)}")
        else:
            logger.info(
                "ℹ️ No KBs attached to this agent — using personality mode only."
            )

        # 3️⃣ Prepare agent context and fallback personality prompt
        agent_name = getattr(req, "agentName", "AI Assistant")
        agent_personality = getattr(
            req, "system_prompt", "You are a friendly and helpful AI assistant."
        )
        agent_intent = getattr(req, "intent", "Assist the user helpfully and clearly.")
        response_config = getattr(req, "responseConfig", {}) or {}

        temperature = response_config.get("temperature", 0.7)
        model = response_config.get("model", "gpt-4o-mini")

        # 4️⃣ If no KB context found → fallback to general personality-driven mode
        if not retrieved_docs:
            logger.info("⚠️ No relevant KB data found — switching to fallback mode.")

            fallback_prompt = f"""
            You are {agent_name}.
            Personality: {agent_personality}
            Intent: {agent_intent}

            Even though no knowledge base context is available, respond naturally, kindly,
            and in a helpful tone. If the user greets you, greet back warmly.

            User query:
            {req.input}
            """

            fallback_completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": agent_personality},
                    {"role": "user", "content": fallback_prompt},
                ],
                temperature=temperature,
            )

            fallback_reply = fallback_completion.choices[0].message.content
            return {"reply": fallback_reply, "mode": "general"}

        # 5️⃣ KB context found → prepare structured prompt
        context = "\n\n".join(retrieved_docs[:10])
        context_prompt = f"""
        You are {agent_name}.
        Personality: {agent_personality}
        Intent: {agent_intent}

        Use the following context from the knowledge base to answer precisely and factually.
        If the answer cannot be found in the context, politely say you don’t have that specific information,
        but still provide a helpful general insight if possible.

        Context:
        {context}

        Question:
        {req.input}
        """

        # 6️⃣ Generate response using OpenAI
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": agent_personality},
                {"role": "user", "content": context_prompt},
            ],
            temperature=temperature,
        )

        reply = completion.choices[0].message.content
        return {"reply": reply, "mode": "kb"}

    except Exception as e:
        logger.exception("❌ QUERY ERROR")
        raise HTTPException(status_code=500, detail=str(e))
