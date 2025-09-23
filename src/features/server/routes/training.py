from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Optional, List
import uuid
import asyncio
from datetime import datetime, timezone

from ...ai.models.models import (
    WebsiteCrawlRequest,
    DocumentUploadRequest,
    FAQTrainingRequest,
    TrainingResponse,
    CrawlStatus,
    MessageResponse,
    MessageRequest,
    TrainingStatus,
    SummaryRequest,
    SupabaseFileTrainingRequest,
    SupabaseDocumentTrainingResponse,
    TrainingJobStatus,
    SimpleFileTrainingRequest,
    SimpleFAQTrainingRequest,
    SimpleTrainingResponse,
    SimpleTrainingJobStatus,
)
from ...ai.vector.vector_store import VectorStoreService
from ...ai.models.models import VectorDocument, Message
from ...ai.services.supabase_service import supabase_service
from ..config import (
    logger,
    API_REQUESTS,
    API_ERRORS,
    API_LATENCY,
    get_default_vector_config,
)
from ..security import authenticate, check_rate_limit

# Create the router for training endpoints
router = APIRouter()

# In-memory storage for crawl status and training jobs (use database in production)
crawl_status_store = {}
training_jobs_store = {}


@router.post("/train", response_model=MessageResponse)
async def train_conversation(
    request: MessageRequest,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit),
):
    """Train a conversation by adding messages to the vector store."""
    try:
        with API_LATENCY.labels("/conversation/train").time():
            API_REQUESTS.labels("/conversation/train").inc()
            logger.info(
                f"TRAINING REQUEST RECEIVED",
                extra={
                    "conversationId": request.conversationId,
                    "message_count": len(request.messages),
                    "userId": request.userId,
                    "locationId": request.locationId,
                    "knowledgebaseId": request.knowledgebaseId,
                    "model_name": getattr(request, "model_name", None),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            logger.debug(
                f"TRAINING - Messages preview: {[{'role': msg.role, 'body_length': len(msg.body or '')} for msg in request.messages[:3]]}"
            )

            knowledgebaseId = request.knowledgebaseId or request.conversationId
            cleaned_messages = []

            for msg in request.messages:
                if not any([msg.body, msg.role, msg.messageType]):
                    continue

                try:
                    msg_data = msg.dict()
                    msg_data.update(
                        {
                            "userId": request.userId,
                            "conversationId": request.conversationId,
                            "locationId": request.locationId or "default",
                            "knowledgebaseId": knowledgebaseId,
                        }
                    )
                    cleaned_messages.append(msg_data)
                except Exception as e:
                    logger.error(f"Error cleaning message data: {str(e)}")
                    continue

            if not cleaned_messages:
                logger.warning(f"TRAINING - No valid messages to train after cleaning")
                raise HTTPException(
                    status_code=400, detail="No valid messages to train"
                )

            logger.info(
                f"TRAINING - Cleaned {len(cleaned_messages)} messages for training"
            )
            logger.debug(
                f"TRAINING - Sample cleaned message: {cleaned_messages[0] if cleaned_messages else 'None'}"
            )

            default_config = get_default_vector_config()
            config = (
                default_config.copy(update={"model_name": request.model_name})
                if hasattr(request, "model_name") and request.model_name
                else default_config
            )
            logger.debug(f"TRAINING - Using vector config: {config}")

            vector_service = VectorStoreService(config=config)
            logger.info(f"TRAINING - Adding messages to vector store...")
            success = await vector_service.add_chat_messages(
                userId=request.userId,
                messages=cleaned_messages,
                knowledgebaseId=knowledgebaseId,
                model_name=(
                    request.model_name if hasattr(request, "model_name") else None
                ),
            )
            logger.info(f"TRAINING - Vector store operation success: {success}")

            now = datetime.now(timezone.utc)
            try:
                start_date = min(
                    [msg.get("dateAdded", now.isoformat()) for msg in cleaned_messages],
                    default=now.isoformat(),
                )
                end_date = max(
                    [msg.get("dateAdded", now.isoformat()) for msg in cleaned_messages],
                    default=now.isoformat(),
                )
            except Exception as e:
                logger.error(f"Error calculating date range: {str(e)}")
                start_date = now.isoformat()
                end_date = now.isoformat()

            response_data = {
                "conversationId": request.conversationId,
                "messageCount": len(cleaned_messages),
                "dateRange": {"start": start_date, "end": end_date},
            }

            return MessageResponse(
                success=success,
                message=(
                    f"Successfully trained {len(cleaned_messages)} messages"
                    if success
                    else "Failed to train conversation"
                ),
                data=response_data,
            )

    except HTTPException as he:
        API_ERRORS.labels("/conversation/train").inc()
        logger.error(f"HTTP error in train: {he.detail}")
        raise
    except Exception as e:
        API_ERRORS.labels("/conversation/train").inc()
        logger.error(f"Error training conversation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error training conversation: {str(e)}"
        )


@router.post("/training-status", response_model=TrainingStatus)
async def check_training_status(
    request: SummaryRequest,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit),
):
    """Check the training status of a conversation in the vector store."""
    try:
        with API_LATENCY.labels("/conversation/training-status").time():
            API_REQUESTS.labels("/conversation/training-status").inc()
            logger.info(
                f"Checking training status for conversation: {request.conversationId}"
            )
            knowledgebaseId = (
                request.filters.get("knowledgebaseId", request.conversationId)
                if request.filters
                else request.conversationId
            )

            try:
                default_config = get_default_vector_config()
                config = (
                    default_config.copy(update={"model_name": request.model_name})
                    if hasattr(request, "model_name") and request.model_name
                    else default_config
                )
                vector_service = VectorStoreService(config=config)
                message_count = await vector_service.get_message_count(
                    userId=request.userId,
                    metadata_filter={
                        "conversationId": request.conversationId,
                        "knowledgebaseId": knowledgebaseId,
                    },
                    knowledgebaseId=knowledgebaseId,
                )
                latest_message = await vector_service.query_chat_history(
                    userId=request.userId,
                    query="",
                    metadata_filter={
                        "conversationId": request.conversationId,
                        "knowledgebaseId": knowledgebaseId,
                    },
                    k=1,
                    knowledgebaseId=knowledgebaseId,
                    model_name=(
                        request.model_name if hasattr(request, "model_name") else None
                    ),
                )
            except Exception as e:
                logger.error(f"Error querying vector store: {str(e)}")
                message_count = 0
                latest_message = []

            is_trained = message_count > 0

            try:
                last_updated = (
                    latest_message[0].get("dateAdded") if latest_message else None
                )
            except Exception as e:
                logger.error(f"Error getting last_updated: {str(e)}")
                last_updated = None

            return TrainingStatus(
                is_trained=is_trained,
                last_updated=last_updated,
                message_count=message_count,
                vector_count=message_count,
            )

    except HTTPException as he:
        API_ERRORS.labels("/conversation/training-status").inc()
        logger.error(f"HTTP error in training-status: {he.detail}")
        raise
    except Exception as e:
        API_ERRORS.labels("/conversation/training-status").inc()
        logger.error(f"Error checking training status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error checking training status: {str(e)}"
        )


@router.post("/training/website", response_model=TrainingResponse)
async def crawl_website(
    request: WebsiteCrawlRequest,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit),
):
    """Start crawling a website for training data."""
    try:
        crawl_id = str(uuid.uuid4())

        # Initialize crawl status
        crawl_status_store[crawl_id] = CrawlStatus(
            crawlId=crawl_id,
            status="pending",
            pagesProcessed=0,
            totalPages=0,
            documentsCreated=0,
        )

        # Start crawling in background
        asyncio.create_task(_crawl_website_background(crawl_id, request))

        return TrainingResponse(
            success=True,
            message=f"Website crawling started. Crawl ID: {crawl_id}",
            documentsProcessed=0,
            vectorsCreated=0,
            knowledgebaseId=request.knowledgebaseId,
        )

    except Exception as e:
        logger.error(f"Failed to start website crawl: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start website crawl: {str(e)}"
        )


async def _crawl_website_background(crawl_id: str, request: WebsiteCrawlRequest):
    """Background task to crawl website."""
    try:
        # Update status to running
        crawl_status_store[crawl_id].status = "running"

        # Simple website crawling logic (in production, use proper crawler like Scrapy)
        import requests
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin, urlparse

        vector_service = VectorStoreService(config=get_default_vector_config())
        visited_urls = set()
        to_visit = [request.url]
        documents_created = 0

        while to_visit and len(visited_urls) < request.maxPages:
            url = to_visit.pop(0)
            if url in visited_urls:
                continue

            try:
                # Check URL patterns
                if request.excludePatterns:
                    if any(pattern in url for pattern in request.excludePatterns):
                        continue

                if request.includePatterns:
                    if not any(pattern in url for pattern in request.includePatterns):
                        continue

                # Fetch page
                response = requests.get(url, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, "html.parser")

                # Extract text content
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text()

                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (
                    phrase.strip() for line in lines for phrase in line.split("  ")
                )
                text = " ".join(chunk for chunk in chunks if chunk)

                if len(text) > 100:  # Only process pages with substantial content
                    # Create a message-like document
                    message = Message(
                        id=str(uuid.uuid4()),
                        body=text[:5000],  # Limit content length
                        direction="inbound",
                        role="web_content",
                        conversationId=f"web_crawl_{crawl_id}",
                        knowledgebaseId=request.knowledgebaseId,
                        userId=request.userId,
                        source=url,
                        contentType="text/html",
                    )

                    # Convert to vector document
                    doc = VectorDocument.from_message(
                        message,
                        request.userId,
                        data_type="web_content",
                        knowledgebaseId=request.knowledgebaseId,
                    )

                    # Store in vector database
                    await vector_service.add_documents([doc], request.knowledgebaseId)
                    documents_created += 1

                visited_urls.add(url)
                crawl_status_store[crawl_id].pagesProcessed = len(visited_urls)
                crawl_status_store[crawl_id].documentsCreated = documents_created

                # Find more URLs to crawl (simple depth-limited crawling)
                if len(visited_urls) < request.maxDepth * 10:  # Simple depth control
                    for link in soup.find_all("a", href=True):
                        new_url = urljoin(url, link["href"])
                        if urlparse(new_url).netloc == urlparse(request.url).netloc:
                            if new_url not in visited_urls and new_url not in to_visit:
                                to_visit.append(new_url)

            except Exception as e:
                logger.warning(f"Failed to crawl {url}: {e}")
                continue

        # Update final status
        crawl_status_store[crawl_id].status = "completed"
        crawl_status_store[crawl_id].completedAt = datetime.now(timezone.utc)
        crawl_status_store[crawl_id].totalPages = len(visited_urls)

        logger.info(
            f"Website crawl {crawl_id} completed: {documents_created} documents created"
        )

    except Exception as e:
        logger.error(f"Website crawl {crawl_id} failed: {e}")
        crawl_status_store[crawl_id].status = "failed"
        crawl_status_store[crawl_id].error = str(e)


@router.get("/training/website/{crawl_id}/status", response_model=CrawlStatus)
async def get_crawl_status(
    crawl_id: str,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit),
):
    """Get the status of a website crawl."""
    if crawl_id not in crawl_status_store:
        raise HTTPException(status_code=404, detail="Crawl not found")

    return crawl_status_store[crawl_id]


# Document upload endpoint removed - use Supabase file training instead
# POST /training/supabase-file or POST /training/supabase-file/async


@router.post("/training/faq", response_model=TrainingResponse)
async def train_faq(
    request: FAQTrainingRequest,
    auth: dict = Depends(authenticate),
    # _rate_limit: None = Depends(check_rate_limit),
):
    """Train with FAQ data."""
    try:
        vector_service = VectorStoreService(config=get_default_vector_config())
        documents_created = 0

        for i, faq in enumerate(request.faqs):
            question = faq.get("question", "").strip()
            answer = faq.get("answer", "").strip()

            if not question or not answer:
                continue

            # Create documents for both question and answer
            faq_content = f"Q: {question}\nA: {answer}"

            message = Message(
                id=str(uuid.uuid4()),
                body=faq_content,
                direction="inbound",
                role="faq_content",
                conversationId=f"faq_{i}",
                knowledgebaseId=request.knowledgebaseId,
                userId=request.userId,
                source="faq_training",
                contentType="text/plain",
            )

            doc = VectorDocument.from_message(
                message,
                request.userId,
                data_type="faq",
                knowledgebaseId=request.knowledgebaseId,
            )

            await vector_service.add_documents([doc], request.knowledgebaseId)
            documents_created += 1
            # ✅ Print the document created
            print(f"[FAQ TRAINING] Document Created: {doc.dict() if hasattr(doc, 'dict') else doc}")

        print(f"[FAQ TRAINING] Total Documents Created: {documents_created}")
            

        return TrainingResponse(
            success=True,
            message=f"FAQ training completed: {documents_created} FAQ items processed",
            documentsProcessed=len(request.faqs),
            vectorsCreated=documents_created,
            knowledgebaseId=request.knowledgebaseId,
        )

    except Exception as e:
        logger.error(f"Failed to train FAQ: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to train FAQ: {str(e)}")


@router.post("/training/supabase-file", response_model=SupabaseDocumentTrainingResponse)
async def train_supabase_file(
    request: SupabaseFileTrainingRequest,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit),
):
    """Train with a file stored in Supabase storage."""
    try:
        start_time = datetime.now(timezone.utc)

        # Download file from Supabase
        try:
            file_content, content_type = await supabase_service.download_file(
                bucket=request.supabaseBucket, file_path=request.fileId
            )
            logger.info(
                f"Downloaded file {request.fileName} ({len(file_content)} bytes)"
            )
        except Exception as e:
            logger.error(f"Failed to download file from Supabase: {e}")
            return SupabaseDocumentTrainingResponse(
                success=False,
                message=f"Failed to download file: {str(e)}",
                fileId=request.fileId,
                fileName=request.fileName,
                documentsProcessed=0,
                vectorsCreated=0,
                knowledgebaseId=request.knowledgebaseId,
                trainingStatus="failed",
            )

        # Extract text content
        try:
            text_content = supabase_service.extract_text_from_content(
                content=file_content,
                content_type=content_type,
                filename=request.fileName,
            )

            if not text_content.strip():
                raise ValueError("No text content found in file")

        except Exception as e:
            logger.error(f"Failed to extract text from file: {e}")
            return SupabaseDocumentTrainingResponse(
                success=False,
                message=f"Failed to extract text: {str(e)}",
                fileId=request.fileId,
                fileName=request.fileName,
                documentsProcessed=0,
                vectorsCreated=0,
                knowledgebaseId=request.knowledgebaseId,
                trainingStatus="failed",
            )

        # Chunk the text
        chunks = supabase_service.chunk_text(text_content)
        logger.info(f"Split file into {len(chunks)} chunks")

        # Create vector documents
        vector_service = VectorStoreService(config=get_default_vector_config())
        documents_created = 0

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 50:  # Only process substantial chunks
                message = Message(
                    id=str(uuid.uuid4()),
                    body=chunk,
                    direction="inbound",
                    role="document_content",
                    conversationId=f"supabase_{request.fileId}_{i}",
                    knowledgebaseId=request.knowledgebaseId,
                    userId=request.userId,
                    source=f"supabase:{request.supabaseBucket}/{request.fileId}",
                    contentType=content_type,
                )

                doc = VectorDocument.from_message(
                    message,
                    request.userId,
                    data_type="supabase_document",
                    knowledgebaseId=request.knowledgebaseId,
                )

                await vector_service.add_documents([doc], request.knowledgebaseId)
                documents_created += 1

        end_time = datetime.now(timezone.utc)
        processing_time = (end_time - start_time).total_seconds()

        return SupabaseDocumentTrainingResponse(
            success=True,
            message=f"Successfully processed file: {documents_created} chunks created",
            fileId=request.fileId,
            fileName=request.fileName,
            documentsProcessed=1,
            vectorsCreated=documents_created,
            knowledgebaseId=request.knowledgebaseId,
            trainingStatus="completed",
            processingTime=processing_time,
        )

    except Exception as e:
        logger.error(f"Failed to train Supabase file: {e}")
        return SupabaseDocumentTrainingResponse(
            success=False,
            message=f"Training failed: {str(e)}",
            fileId=request.fileId,
            fileName=request.fileName,
            documentsProcessed=0,
            vectorsCreated=0,
            knowledgebaseId=request.knowledgebaseId,
            trainingStatus="failed",
        )


@router.post("/training/supabase-file/async", response_model=TrainingJobStatus)
async def start_supabase_file_training_job(
    request: SupabaseFileTrainingRequest,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit),
):
    """Start an async training job for a Supabase file."""
    try:
        job_id = str(uuid.uuid4())

        # Create job status
        job_status = TrainingJobStatus(
            jobId=job_id,
            userId=request.userId,
            knowledgebaseId=request.knowledgebaseId,
            fileId=request.fileId,
            fileName=request.fileName,
            status="pending",
            metadata=request.metadata or {},
        )

        training_jobs_store[job_id] = job_status

        # Start background processing
        asyncio.create_task(_process_supabase_file_background(job_id, request))

        return job_status

    except Exception as e:
        logger.error(f"Failed to start Supabase file training job: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start training job: {str(e)}"
        )


async def _process_supabase_file_background(
    job_id: str, request: SupabaseFileTrainingRequest
):
    """Background task to process Supabase file training."""
    try:
        # Update status to processing
        training_jobs_store[job_id].status = "processing"
        training_jobs_store[job_id].progress = 0.1

        # Download file
        file_content, content_type = await supabase_service.download_file(
            bucket=request.supabaseBucket, file_path=request.fileId
        )
        training_jobs_store[job_id].progress = 0.3

        # Extract text
        text_content = supabase_service.extract_text_from_content(
            content=file_content, content_type=content_type, filename=request.fileName
        )
        training_jobs_store[job_id].progress = 0.5

        # Chunk text
        chunks = supabase_service.chunk_text(text_content)
        training_jobs_store[job_id].progress = 0.6

        # Process chunks
        vector_service = VectorStoreService(config=get_default_vector_config())
        documents_created = 0

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 50:
                message = Message(
                    id=str(uuid.uuid4()),
                    body=chunk,
                    direction="inbound",
                    role="document_content",
                    conversationId=f"supabase_{request.fileId}_{i}",
                    knowledgebaseId=request.knowledgebaseId,
                    userId=request.userId,
                    source=f"supabase:{request.supabaseBucket}/{request.fileId}",
                    contentType=content_type,
                )

                doc = VectorDocument.from_message(
                    message,
                    request.userId,
                    data_type="supabase_document",
                    knowledgebaseId=request.knowledgebaseId,
                )

                await vector_service.add_documents([doc], request.knowledgebaseId)
                documents_created += 1

                # Update progress
                progress = 0.6 + (0.4 * (i + 1) / len(chunks))
                training_jobs_store[job_id].progress = min(progress, 1.0)

        # Complete job
        training_jobs_store[job_id].status = "completed"
        training_jobs_store[job_id].progress = 1.0
        training_jobs_store[job_id].documentsProcessed = 1
        training_jobs_store[job_id].vectorsCreated = documents_created
        training_jobs_store[job_id].completedAt = datetime.now(timezone.utc)

        logger.info(
            f"Supabase file training job {job_id} completed: {documents_created} vectors created"
        )

    except Exception as e:
        logger.error(f"Supabase file training job {job_id} failed: {e}")
        training_jobs_store[job_id].status = "failed"
        training_jobs_store[job_id].error = str(e)


@router.get("/training/job/{job_id}/status", response_model=TrainingJobStatus)
async def get_training_job_status(
    job_id: str,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit),
):
    """Get the status of a training job."""
    if job_id not in training_jobs_store:
        raise HTTPException(status_code=404, detail="Training job not found")

    return training_jobs_store[job_id]


# ============================================================================
# SIMPLIFIED TRAINING ENDPOINTS (MAIN ENDPOINTS TO USE)
# ============================================================================


@router.post("/training/file", response_model=SimpleTrainingResponse)
async def train_file(
    request: SimpleFileTrainingRequest,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit),
):
    """
    Train AI with a file from Supabase storage.
    Only requires userId and fileId - all metadata retrieved automatically.
    The fileId becomes the knowledgebaseId for easy reference.
    """
    try:
        start_time = datetime.now(timezone.utc)

        # Use fileId as knowledgebaseId
        knowledgebase_id = request.fileId

        # Get file info and download automatically
        try:
            file_content, content_type, file_info = (
                await supabase_service.download_file_by_id(request.fileId)
            )
            logger.info(
                f"Downloaded file {file_info['fileName']} ({len(file_content)} bytes)"
            )
        except Exception as e:
            logger.error(f"Failed to download file {request.fileId}: {e}")
            return SimpleTrainingResponse(
                success=False,
                message=f"Failed to download file: {str(e)}",
                knowledgebaseId=knowledgebase_id,
                documentsProcessed=0,
                vectorsCreated=0,
            )

        # Extract text content
        try:
            text_content = supabase_service.extract_text_from_content(
                content=file_content,
                content_type=content_type,
                filename=file_info["fileName"],
            )

            if not text_content.strip():
                raise ValueError("No text content found in file")

        except Exception as e:
            logger.error(f"Failed to extract text from file: {e}")
            return SimpleTrainingResponse(
                success=False,
                message=f"Failed to extract text: {str(e)}",
                knowledgebaseId=knowledgebase_id,
                documentsProcessed=0,
                vectorsCreated=0,
            )

        # Chunk the text
        chunks = supabase_service.chunk_text(text_content)
        logger.info(f"Split file into {len(chunks)} chunks")

        # Create vector documents
        vector_service = VectorStoreService(config=get_default_vector_config())
        documents_created = 0

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 50:  # Only process substantial chunks
                message = Message(
                    id=str(uuid.uuid4()),
                    body=chunk,
                    direction="inbound",
                    role="document_content",
                    conversationId=f"file_{request.fileId}_{i}",
                    knowledgebaseId=knowledgebase_id,
                    userId=request.userId,
                    source=f"file:{request.fileId}",
                    contentType=content_type,
                )

                doc = VectorDocument.from_message(
                    message,
                    request.userId,
                    data_type="file_document",
                    knowledgebaseId=knowledgebase_id,
                )

                await vector_service.add_documents([doc], knowledgebase_id)
                documents_created += 1

        end_time = datetime.now(timezone.utc)
        processing_time = (end_time - start_time).total_seconds()

        return SimpleTrainingResponse(
            success=True,
            message=f"Successfully processed file '{file_info['fileName']}': {documents_created} chunks created",
            knowledgebaseId=knowledgebase_id,
            documentsProcessed=1,
            vectorsCreated=documents_created,
            processingTime=processing_time,
        )

    except Exception as e:
        logger.error(f"Failed to train file {request.fileId}: {e}")
        return SimpleTrainingResponse(
            success=False,
            message=f"Training failed: {str(e)}",
            knowledgebaseId=request.fileId,
            documentsProcessed=0,
            vectorsCreated=0,
        )


@router.post("/training/file/async", response_model=SimpleTrainingJobStatus)
async def train_file_async(
    request: SimpleFileTrainingRequest,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit),
):
    """
    Start async training job for a file.
    Only requires userId and fileId.
    """
    try:
        job_id = str(uuid.uuid4())
        knowledgebase_id = request.fileId

        # Create job status
        job_status = SimpleTrainingJobStatus(
            jobId=job_id,
            userId=request.userId,
            knowledgebaseId=knowledgebase_id,
            fileId=request.fileId,
            status="pending",
        )

        training_jobs_store[job_id] = job_status

        # Start background processing
        asyncio.create_task(_process_file_background(job_id, request))

        return job_status

    except Exception as e:
        logger.error(f"Failed to start file training job: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start training job: {str(e)}"
        )


@router.post("/training/faq", response_model=SimpleTrainingResponse)
async def train_faq(
    request: SimpleFAQTrainingRequest,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit),
):
    """
    Train AI with FAQ data.
    Only requires userId and faqs array. KnowledgebaseId is auto-generated if not provided.
    """
    try:
        start_time = datetime.now(timezone.utc)

        # Generate knowledgebaseId if not provided
        knowledgebase_id = (
            request.knowledgebaseId
            or f"faq_{request.userId}_{int(datetime.now().timestamp())}"
        )

        vector_service = VectorStoreService(config=get_default_vector_config())
        documents_created = 0

        for i, faq in enumerate(request.faqs):
            question = faq.get("question", "").strip()
            answer = faq.get("answer", "").strip()

            if not question or not answer:
                continue

            # Create documents for both question and answer
            faq_content = f"Q: {question}\nA: {answer}"

            message = Message(
                id=str(uuid.uuid4()),
                body=faq_content,
                direction="inbound",
                role="faq_content",
                conversationId=f"faq_{knowledgebase_id}_{i}",
                knowledgebaseId=knowledgebase_id,
                userId=request.userId,
                source="faq_training",
                contentType="text/plain",
            )

            doc = VectorDocument.from_message(
                message,
                request.userId,
                data_type="faq",
                knowledgebaseId=knowledgebase_id,
            )

            await vector_service.add_documents([doc], knowledgebase_id)
            documents_created += 1

        end_time = datetime.now(timezone.utc)
        processing_time = (end_time - start_time).total_seconds()

        return SimpleTrainingResponse(
            success=True,
            message=f"FAQ training completed: {documents_created} FAQ items processed",
            knowledgebaseId=knowledgebase_id,
            documentsProcessed=len(request.faqs),
            vectorsCreated=documents_created,
            processingTime=processing_time,
        )

    except Exception as e:
        logger.error(f"Failed to train FAQ: {e}")
        return SimpleTrainingResponse(
            success=False,
            message=f"FAQ training failed: {str(e)}",
            knowledgebaseId=request.knowledgebaseId or "unknown",
            documentsProcessed=0,
            vectorsCreated=0,
        )


@router.get("/training/job/{job_id}/status", response_model=SimpleTrainingJobStatus)
async def get_training_status(
    job_id: str,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit),
):
    """Get the status of a training job."""
    if job_id not in training_jobs_store:
        raise HTTPException(status_code=404, detail="Training job not found")

    return training_jobs_store[job_id]


async def _process_file_background(job_id: str, request: SimpleFileTrainingRequest):
    """Background task to process file training."""
    try:
        knowledgebase_id = request.fileId

        # Update status to processing
        training_jobs_store[job_id].status = "processing"
        training_jobs_store[job_id].progress = 0.1

        # Get file info and download
        file_content, content_type, file_info = (
            await supabase_service.download_file_by_id(request.fileId)
        )
        training_jobs_store[job_id].progress = 0.3

        # Extract text
        text_content = supabase_service.extract_text_from_content(
            content=file_content,
            content_type=content_type,
            filename=file_info["fileName"],
        )
        training_jobs_store[job_id].progress = 0.5

        # Chunk text
        chunks = supabase_service.chunk_text(text_content)
        training_jobs_store[job_id].progress = 0.6

        # Process chunks
        vector_service = VectorStoreService(config=get_default_vector_config())
        documents_created = 0

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 50:
                message = Message(
                    id=str(uuid.uuid4()),
                    body=chunk,
                    direction="inbound",
                    role="document_content",
                    conversationId=f"file_{request.fileId}_{i}",
                    knowledgebaseId=knowledgebase_id,
                    userId=request.userId,
                    source=f"file:{request.fileId}",
                    contentType=content_type,
                )

                doc = VectorDocument.from_message(
                    message,
                    request.userId,
                    data_type="file_document",
                    knowledgebaseId=knowledgebase_id,
                )

                await vector_service.add_documents([doc], knowledgebase_id)
                documents_created += 1

                # Update progress
                progress = 0.6 + (0.4 * (i + 1) / len(chunks))
                training_jobs_store[job_id].progress = min(progress, 1.0)

        # Complete job
        training_jobs_store[job_id].status = "completed"
        training_jobs_store[job_id].progress = 1.0
        training_jobs_store[job_id].documentsProcessed = 1
        training_jobs_store[job_id].vectorsCreated = documents_created
        training_jobs_store[job_id].completedAt = datetime.now(timezone.utc)

        logger.info(
            f"File training job {job_id} completed: {documents_created} vectors created"
        )

    except Exception as e:
        logger.error(f"File training job {job_id} failed: {e}")
        training_jobs_store[job_id].status = "failed"
        training_jobs_store[job_id].error = str(e)


# ============================================================================
# LEGACY ENDPOINTS (KEPT FOR BACKWARD COMPATIBILITY)
# ============================================================================


@router.post("/training/faq/legacy", response_model=TrainingResponse)
async def train_faq_legacy(
    request: FAQTrainingRequest,
    auth: dict = Depends(authenticate),
    _rate_limit: None = Depends(check_rate_limit),
):
    """
    DEPRECATED: Use /training/faq instead.
    Legacy FAQ training endpoint with complex parameters.
    """
    try:
        vector_service = VectorStoreService(config=get_default_vector_config())
        documents_created = 0

        for i, faq in enumerate(request.faqs):
            question = faq.get("question", "").strip()
            answer = faq.get("answer", "").strip()

            if not question or not answer:
                continue

            # Create documents for both question and answer
            faq_content = f"Q: {question}\nA: {answer}"

            message = Message(
                id=str(uuid.uuid4()),
                body=faq_content,
                direction="inbound",
                role="faq_content",
                conversationId=f"faq_{i}",
                knowledgebaseId=request.knowledgebaseId,
                userId=request.userId,
                source="faq_training",
                contentType="text/plain",
            )

            doc = VectorDocument.from_message(
                message,
                request.userId,
                data_type="faq",
                knowledgebaseId=request.knowledgebaseId,
            )

            await vector_service.add_documents([doc], request.knowledgebaseId)
            documents_created += 1

        return TrainingResponse(
            success=True,
            message=f"FAQ training completed: {documents_created} FAQ items processed",
            documentsProcessed=len(request.faqs),
            vectorsCreated=documents_created,
            knowledgebaseId=request.knowledgebaseId,
        )

    except Exception as e:
        logger.error(f"Failed to train FAQ: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to train FAQ: {str(e)}")


# Note: Removed complex Supabase endpoints - use simplified /training/file instead
