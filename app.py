
# ============================================================
# IMPORTS AND ENVIRONMENT CONFIGURATION
# ============================================================

from dotenv import load_dotenv
import os
import certifi

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import json
import uuid
import time
import logging
from pathlib import Path
from functools import lru_cache

import uvicorn

from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    File,
    Form
)

from fastapi.responses import (
    StreamingResponse,
    JSONResponse
)

from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage
)

from agent import get_agent

from database import (
    init_db,
    save_chat_message,
    get_chat_history,
    create_or_update_conversation,
    list_conversations
)

from rag import add_document_to_rag
from tools import set_current_thread_id


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("Agentic_RAG_Chatbot")


# ============================================================
# FASTAPI INITIALIZATION
# ============================================================

app = FastAPI(
    title="Agentic_RAG_Chatbot",
    description="Agentic AI chatbot with RAG, memory, tools and Gemini",
    version="1.0.0"
)

templates = Jinja2Templates(directory="templates")

Path("uploads").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

init_db()


# ============================================================
# CACHE AGENT INSTANCES
# ============================================================

# Reuse agents for the same model rather than compiling
# the LangGraph workflow for every message.
#
# This assumes get_agent() returns a reusable graph with
# thread-isolated state and a reusable checkpointer.

@lru_cache(maxsize=5)
def get_cached_agent(model_name: str):
    logger.info("Initializing agent for model: %s", model_name)

    start = time.perf_counter()

    agent = get_agent(model_name)

    logger.info(
        "Agent initialized in %.3f seconds",
        time.perf_counter() - start
    )

    return agent


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# ============================================================
# LIST CONVERSATIONS
# ============================================================

@app.get("/conversations")
async def conversations():

    items = await run_in_threadpool(list_conversations)

    return {
        "conversations": [
            {
                "thread_id": item.thread_id,
                "title": item.title,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat()
            }
            for item in items
        ]
    }


# ============================================================
# GET CONVERSATION HISTORY
# ============================================================

@app.get("/history/{thread_id}")
async def history(thread_id: str):

    messages = await run_in_threadpool(
        get_chat_history,
        thread_id
    )

    return {
        "messages": [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
    }


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    thread_id: str = Form(...)
):

    start = time.perf_counter()

    try:
        allowed_extensions = {
            ".pdf",
            ".docx",
            ".txt",
            ".md",
            ".py",
            ".csv"
        }

        filename = Path(
            file.filename or "uploaded_file"
        ).name

        suffix = Path(filename).suffix.lower()

        if suffix not in allowed_extensions:
            return JSONResponse(
                {
                    "success": False,
                    "message": (
                        "Unsupported file type. Upload PDF, "
                        "DOCX, TXT, MD, PY, or CSV."
                    )
                },
                status_code=400
            )

        file_id = str(uuid.uuid4())

        safe_filename = filename.replace(" ", "_")

        file_path = str(
            Path("uploads") /
            f"{file_id}_{safe_filename}"
        )

        # Save uploaded file in chunks.
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                await run_in_threadpool(
                    f.write,
                    chunk
                )

        await run_in_threadpool(
            create_or_update_conversation,
            thread_id,
            "Uploaded document"
        )

        # RAG indexing is synchronous and can be expensive.
        # Run it in a worker thread rather than blocking FastAPI.
        result = await run_in_threadpool(
            add_document_to_rag,
            file_path=file_path,
            thread_id=thread_id
        )

        logger.info(
            "Document uploaded and indexed in %.3fs",
            time.perf_counter() - start
        )

        return JSONResponse(
            {
                "success": True,
                "message": (
                    f"Uploaded {result['filename']} and "
                    f"created {result['chunks']} chunks."
                )
            }
        )

    except Exception as e:

        logger.exception("Document upload failed")

        return JSONResponse(
            {
                "success": False,
                "message": str(e)
            },
            status_code=500
        )

    finally:
        await file.close()


# ============================================================
# SERVER-SENT EVENT FORMAT
# ============================================================

def sse_data(payload: dict) -> str:

    return (
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    )


# ============================================================
# FILTER TOOL OUTPUTS FROM CHAT STREAM
# ============================================================

def should_stream_chunk(chunk, metadata) -> bool:

    """
    Stream normal AI text only.

    Ignore:
    - ToolMessage
    - LangGraph tool nodes
    - Tool-call chunks
    - Invalid tool calls
    - Raw search/RAG tool outputs
    """

    metadata = metadata or {}

    node_name = str(
        metadata.get("langgraph_node", "")
    ).lower()

    if "tool" in node_name:
        return False

    if isinstance(chunk, ToolMessage):
        return False

    if not isinstance(
        chunk,
        (AIMessage, AIMessageChunk)
    ):
        return False

    if getattr(chunk, "tool_calls", None):
        return False

    if getattr(chunk, "invalid_tool_calls", None):
        return False

    additional_kwargs = (
        getattr(chunk, "additional_kwargs", {}) or {}
    )

    if additional_kwargs.get("tool_calls"):
        return False

    return True


# ============================================================
# EXTRACT TEXT FROM MODEL CHUNKS
# ============================================================

def extract_text_from_chunk(chunk) -> str:

    content = getattr(chunk, "content", "")

    if not content:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):

                item_type = item.get("type")

                # Avoid displaying thinking/reasoning blocks.
                if item_type not in (None, "text"):
                    continue

                if isinstance(item.get("text"), str):
                    text_parts.append(item["text"])

                elif isinstance(item.get("content"), str):
                    text_parts.append(item["content"])

        return "".join(text_parts)

    return ""


# ============================================================
# CHAT STREAMING ENDPOINT
# ============================================================

@app.post("/chat/stream")
async def chat_stream(request: Request):

    request_start = time.perf_counter()

    # --------------------------------------------------------
    # READ REQUEST DATA
    # --------------------------------------------------------

    try:
        data = await request.json()

    except Exception:
        return JSONResponse(
            {"error": "Invalid JSON body."},
            status_code=400
        )

    user_message = data.get("message", "")
    thread_id = data.get("thread_id", "default")
    selected_model = data.get(
        "model",
        "gemini-3.8-flash-lite"
    )

    if (
        not isinstance(user_message, str)
        or not user_message.strip()
    ):
        return JSONResponse(
            {"error": "Message is required."},
            status_code=400
        )

    if not isinstance(thread_id, str) or not thread_id.strip():
        return JSONResponse(
            {"error": "Valid thread ID is required."},
            status_code=400
        )

    if not isinstance(selected_model, str):
        return JSONResponse(
            {"error": "Valid model is required."},
            status_code=400
        )

    # Keep model choices aligned with your frontend/backend.
    ALLOWED_MODELS = {
        "gemini-4-hydrogen",
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
    }

    if selected_model not in ALLOWED_MODELS:
        return JSONResponse(
            {"error": "Unsupported model selected."},
            status_code=400
        )

    # --------------------------------------------------------
    # AGENT INITIALIZATION
    # --------------------------------------------------------

    agent_start = time.perf_counter()

    try:
        agent = await run_in_threadpool(
            get_cached_agent,
            selected_model
        )

    except Exception as e:
        logger.exception("Agent initialization failed")

        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

    logger.info(
        "[PERFORMANCE] Agent retrieval: %.3fs",
        time.perf_counter() - agent_start
    )

    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    db_start = time.perf_counter()

    try:
        await run_in_threadpool(
            create_or_update_conversation,
            thread_id,
            user_message
        )

        await run_in_threadpool(
            save_chat_message,
            thread_id,
            "user",
            user_message
        )

    except Exception:
        logger.exception("Failed to save user message")

        return JSONResponse(
            {"error": "Could not save conversation."},
            status_code=500
        )

    logger.info(
        "[PERFORMANCE] Database operations: %.3fs",
        time.perf_counter() - db_start
    )

    # Existing thread handling is preserved.
    # Note: this global setter is only safe for sequential use.
    set_current_thread_id(thread_id)

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    logger.info(
        "[PERFORMANCE] Stream setup: %.3fs",
        time.perf_counter() - request_start
    )

    # --------------------------------------------------------
    # STREAM GENERATOR
    # --------------------------------------------------------

    def event_generator():

        final_answer = ""
        stream_start = time.perf_counter()

        first_token_received = False
        chunk_count = 0

        try:

            inputs = {
                "messages": [
                    HumanMessage(content=user_message)
                ]
            }

            logger.info(
                "[PERFORMANCE] Starting LangGraph stream"
            )

            # Initial SSE event makes streaming state visible
            # to clients before the first Gemini text token.
            yield sse_data({"status": "thinking"})

            for chunk, metadata in agent.stream(
                inputs,
                config=config,
                stream_mode="messages"
            ):

                if not should_stream_chunk(
                    chunk,
                    metadata
                ):
                    continue

                token = extract_text_from_chunk(chunk)

                if not token:
                    continue

                if not first_token_received:

                    first_token_received = True

                    first_token_time = (
                        time.perf_counter() - stream_start
                    )

                    logger.info(
                        "[PERFORMANCE] First visible token: %.3fs",
                        first_token_time
                    )

                chunk_count += 1
                final_answer += token

                yield sse_data({
                    "token": token
                })

            generation_time = (
                time.perf_counter() - stream_start
            )

            logger.info(
                "[PERFORMANCE] Agent execution: %.3fs",
                generation_time
            )

            logger.info(
                "[PERFORMANCE] Streamed chunks: %s",
                chunk_count
            )

            # ------------------------------------------------
            # SAVE ASSISTANT RESPONSE
            # ------------------------------------------------

            if final_answer.strip():

                save_start = time.perf_counter()

                save_chat_message(
                    thread_id,
                    "assistant",
                    final_answer
                )

                logger.info(
                    "[PERFORMANCE] Save response: %.3fs",
                    time.perf_counter() - save_start
                )

            logger.info(
                "[PERFORMANCE] Total request: %.3fs",
                time.perf_counter() - request_start
            )

            yield sse_data({
                "done": True
            })

        except Exception as e:

            logger.exception("Streaming failed")

            yield sse_data({
                "error": str(e)
            })

            yield sse_data({
                "done": True
            })

    # --------------------------------------------------------
    # RETURN STREAMING RESPONSE
    # --------------------------------------------------------

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
