from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from ...ai.services.supabase_service import supabase_service
from ..config import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import openai
import chromadb
from chromadb.config import Settings

# Create the router for endpoints
router = APIRouter()

openai.api_key = os.getenv("OPENAI_API_KEY")

# reuse your existing Chroma client (or create if none)
chroma_client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./chroma_db",  # or your remote URL
    )
)


class FileTrainingRequest(BaseModel):
    userId: str
    knowledgebaseId: str
    fileId: str  # Supabase storage path  kb/{id}/raw/...
    fileName: str
    fileType: str
    fileSize: int
    storagePath: str
    supabaseBucket: str


@router.post("/kb/add-files")
async def train_supabase_file(
    body: FileTrainingRequest, authorization: str = Header(...)
):
    """
    Receives file metadata + bucket path.
    TODO: download → chunk → embed → Chroma → update status
    """

    # Download file from supabase
    try:
        file_content, _ = await supabase_service.download_file(
            bucket=body.supabaseBucket, file_path=body.storagePath
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"download failed: {e}")

    text = file_content.decode("utf-8")  # or use your parser if PDF

    # 1. chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    # 2. embed + store
    collection = chroma_client.get_or_create_collection(name=body.knowledgebaseId)
    for idx, chunk in enumerate(chunks):
        emb = openai.Embedding.create(input=chunk, model="text-embedding-ada-002")[
            "data"
        ][0]["embedding"]
        collection.add(
            documents=[chunk],
            embeddings=[emb],
            metadatas=[{"source": body.fileName, "chunk": idx}],
            ids=[f"{body.fileId}_{idx}"],
        )

    return {"success": True, "jobId": "job_" + body.knowledgebaseId}
