import hashlib
import io
import os
import re
import time
from typing import List

import chromadb
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse 
from fastapi.staticfiles import StaticFiles
from minio import Minio
import fitz
import uvicorn


# ============================================================
# Configuration
# ============================================================

MINIO_ENDPOINT = os.getenv(
    "MINIO_ENDPOINT",
    "minio.tiny-agent.svc.cluster.local:9000"
)

MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

MINIO_BUCKET = os.getenv("MINIO_BUCKET", "knowledge")

CHROMA_HOST = os.getenv(
    "CHROMA_HOST",
    "chroma.tiny-agent.svc.cluster.local"
)

CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://ollama.tiny-agent.svc.cluster.local:11434"
)

EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "qwen3-embedding:0.6b"
)

CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    "qwen3:1.7b"
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
NUM_PREDICT = 124

OLLAMA_TIMEOUT = 180


# ============================================================
# Clients
# ============================================================

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

chroma_client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
)

collection = chroma_client.get_or_create_collection(
    name="knowledge"
)

app = FastAPI(title="Tiny Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://c8-labs.makelabs.in",
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Serve static files 
app.mount( "/static", StaticFiles(directory="static"), name="static" )

# ============================================================
# MinIO
# ============================================================

def ensure_bucket():
    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)


def list_objects():
    return list(
        minio_client.list_objects(
            MINIO_BUCKET,
            recursive=True
        )
    )


def read_object(object_name: str) -> bytes:
    response = minio_client.get_object(
        MINIO_BUCKET,
        object_name
    )

    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


# ============================================================
# Document extraction
# ============================================================

def extract_document(object_name: str, data: bytes) -> str:
    name = object_name.lower()

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="replace")

    if name.endswith(".pdf"):
        return extract_pdf(data)

    raise ValueError(
        f"Unsupported document type: {object_name}"
    )


def extract_pdf(data: bytes) -> str:
    document = fitz.open(
        stream=data,
        filetype="pdf"
    )

    pages = []

    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")

            if text.strip():
                pages.append(
                    f"[Page {page_number}]\n{text}"
                )
    finally:
        document.close()

    return "\n\n".join(pages)


# ============================================================
# Chunking
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


def chunk_text_by_paragraph(
    text: str,
    chunk_size: int = CHUNK_SIZE,
) -> List[str]:

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    if not text:
        return []

    chunks = []

    for paragraph in re.split(r"\n[ \t]*\n+", text):
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(paragraph) > chunk_size:
            midpoint = len(paragraph) // 2
            split_at = paragraph.rfind(" ", 0, midpoint)

            if split_at <= 0:
                split_at = midpoint

            chunks.extend((
                paragraph[:split_at].strip(),
                paragraph[split_at:].strip(),
            ))
        else:
            chunks.append(paragraph)

    return chunks


# ============================================================
# Embeddings
# ============================================================

def embed(text: str) -> List[float]:

    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": text,
        },
        timeout=OLLAMA_TIMEOUT,
    )

    response.raise_for_status()

    payload = response.json()

    embeddings = payload.get("embeddings")

    if not embeddings:
        raise RuntimeError(
            "Ollama returned no embeddings"
        )

    return embeddings[0]


# ============================================================
# Chroma helpers
# ============================================================

def make_chunk_id(
    object_name: str,
    chunk_number: int
) -> str:

    value = f"{object_name}:{chunk_number}"

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def delete_source_chunks(object_name: str):

    existing = collection.get(
        where={
            "source": object_name
        },
        include=[]
    )

    ids = existing.get("ids", [])

    if ids:
        collection.delete(
            ids=ids
        )

        print(
            f"Deleted {len(ids)} old chunks "
            f"for {object_name}"
        )


# ============================================================
# Ingestion
# ============================================================

def ingest_object(obj):

    object_name = obj.object_name

    print(f"Reading: {object_name}")

    data = read_object(object_name)

    print(
        f"Bytes: {len(data)}"
    )

    text = extract_document(
        object_name,
        data
    )

    if object_name.lower().endswith(".txt"):
        chunks = chunk_text_by_paragraph(text)
    else:
        chunks = chunk_text(text)

    if not chunks:
        print(
            f"No text extracted from {object_name}"
        )
        return 0

    # Replace existing chunks for this source.
    delete_source_chunks(object_name)

    count = 0

    for chunk_number, chunk in enumerate(chunks):

        print(
            f"Embedding "
            f"{object_name} chunk {chunk_number}"
        )

        vector = embed(chunk)

        chunk_id = make_chunk_id(
            object_name,
            chunk_number
        )

        metadata = {
            "source": object_name,
            "chunk": chunk_number,
            "etag": obj.etag,
        }

        # Do this one chunk at a time.
        # It makes the prototype easier to reason about
        # and avoids list-length/API ambiguity.
        collection.upsert(
            ids=[chunk_id],
            documents=[chunk],
            embeddings=[vector],
            metadatas=[metadata],
        )

        count += 1

    print(
        f"Indexed {count} chunks from {object_name}"
    )

    return count


def ingest_all(reset: bool = False):

    global collection

    ensure_bucket()

    if reset:
        print("RESET requested")

        try:
            chroma_client.delete_collection(
                "knowledge"
            )
        except Exception:
            pass

        collection = chroma_client.get_or_create_collection(
            name="knowledge"
        )

        print("Chroma collection recreated")

    objects = list_objects()

    print(
        f"Objects found: {len(objects)}"
    )

    total = 0

    for obj in objects:

        # Ignore directory placeholders.
        if not obj.object_name:
            continue

        try:
            total += ingest_object(obj)

        except Exception as exc:
            print(
                f"ERROR ingesting "
                f"{obj.object_name}: {exc}"
            )

    print(
        f"Collection count: "
        f"{collection.count()}"
    )

    return {
        "objects": len(objects),
        "chunks_indexed": total,
        "collection_count": collection.count(),
    }


# ============================================================
# Retrieval
# ============================================================

def retrieve(
    query: str,
    n: int = 5
):

    query_vector = embed(query)

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=n,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = result.get(
        "documents",
        [[]]
    )[0]

    metadatas = result.get(
        "metadatas",
        [[]]
    )[0]

    distances = result.get(
        "distances",
        [[]]
    )[0]

    results = []

    for i, document in enumerate(documents):

        metadata = (
            metadatas[i]
            if i < len(metadatas)
            else {}
        )

        distance = (
            distances[i]
            if i < len(distances)
            else None
        )

        results.append({
            "text": document,
            "metadata": metadata,
            "distance": distance,
        })

    return results


# ============================================================
# Context
# ============================================================

def build_context(results):

    parts = []

    for index, result in enumerate(results, start=1):

        metadata = result["metadata"]

        source = metadata.get(
            "source",
            "unknown"
        )

        chunk = metadata.get(
            "chunk",
            0
        )

        text = result["text"]

        parts.append(
            f"[SOURCE {index}]\n"
            f"Content:\n{text}"
        )

    return "\n\n".join(parts)


# ============================================================
# Answer generation
# ============================================================

def generate_answer(
    question: str,
    context: str,
):

    system_prompt = """
Answer using only the supplied context. Do not infer or add facts.
If the context is insufficient, say so. Cite factual claims as [SOURCE N]. 
Be concise.
""".strip()

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "stream": False,
            "think": False,
            "options": {
                "num_predict": NUM_PREDICT,
                "temperature": 0.1
            },
        },
        timeout=OLLAMA_TIMEOUT,
    )

    response.raise_for_status()

    payload = response.json()

    message = payload.get("message", {})

    answer = message.get(
        "content",
        ""
    ).strip()

    if not answer:
        raise RuntimeError(
            "Ollama returned an empty answer"
        )

    return answer


# ============================================================
# API
# ============================================================

# Render the web UI 
@app.get("/") 
def home(): 
    return FileResponse("static/index.html")

@app.get("/health")
def health():

    try:
        heartbeat = chroma_client.heartbeat()

        return {
            "status": "ok",
            "chroma": heartbeat,
            "collection_count": collection.count(),
        }

    except Exception as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )


@app.post("/ingest")
def ingest_endpoint(
    reset: bool = False
):

    try:

        result = ingest_all(
            reset=reset
        )

        return {
            "status": "ok",
            **result,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


@app.get("/search")
def search(
    q: str,
    n: int = 5,
):

    if not q.strip():
        raise HTTPException(
            status_code=400,
            detail="q is required"
        )

    try:

        results = retrieve(
            q,
            n
        )

        return {
            "query": q,
            "results": results,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


@app.get("/ask")
def ask(
    q: str,
    n: int = 5,
):

    if not q.strip():
        raise HTTPException(
            status_code=400,
            detail="q is required"
        )

    try:

        results = retrieve(
            q,
            n
        )

        if not results:

            return {
                "query": q,
                "answer": (
                    "The knowledge base does not "
                    "contain enough information "
                    "to answer this question."
                ),
                "sources": [],
            }

        context = build_context(
            results
        )

        answer = generate_answer(
            q,
            context
        )

        sources = []

        for result in results:

            metadata = result["metadata"]

            sources.append({
                "source": metadata.get(
                    "source"
                ),
                "chunk": metadata.get(
                    "chunk"
                ),
                "distance": result.get(
                    "distance"
                ),
            })

        return {
            "query": q,
            "answer": answer,
            "sources": sources,
        }

    except requests.RequestException as exc:

        raise HTTPException(
            status_code=502,
            detail=f"Ollama request failed: {exc}"
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("Starting Tiny Agent API")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
    )