from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, HttpUrl
from src.features.server.sockets.sockets import sio_server, connected_clients
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI
import chromadb
import os
from src.features.ai.agents.custom_agent_service import logger
import aiohttp, bs4, re
from urllib.parse import urljoin

router = APIRouter()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.Client()


class FileTrainingRequest(BaseModel):
    userId: str
    knowledgebaseId: str
    fileId: str
    fileName: str
    fileType: str
    fileSize: int
    storagePath: str
    supabaseBucket: str


class WebCrawlRequest(BaseModel):
    userId: str
    knowledgebaseId: str
    url: HttpUrl


class FaqTrainingRequest(BaseModel):
    userId: str
    knowledgebaseId: str
    faqs: list[dict]


async def emit_progress(user_id: str, event: str, payload: dict):
    """Helper to safely emit to a user's socket if connected."""
    sid = connected_clients.get(user_id)
    if sid:
        await sio_server.emit(event, payload, room=sid)
    else:
        print(f"⚠️ No active socket for user {user_id}")


@router.post("/kb/add-files")
async def train_supabase_file(
    body: FileTrainingRequest, authorization: str = Header(...)
):
    try:
        # 0️⃣ Emit start
        await emit_progress(
            body.userId,
            "file_status",
            {
                "status": "starting",
                "message": f"Processing {body.fileName}...",
                "fileName": body.fileName,
            },
        )

        # 1️⃣ Chunking
        text = "dummy text for demo"  # replace with actual file content
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
