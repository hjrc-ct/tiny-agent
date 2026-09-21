import hashlib
import io
import os

import chromadb
import pymupdf
import requests

from fastapi import FastAPI, Query
from minio import Minio
import uvicorn


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MINIO_ENDPOINT = "minio.tiny-agent.svc.cluster.local:9000"
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]

BUCKET = "knowledge"

OLLAMA_URL = "http://ollama.tiny-agent.svc.cluster.local:11434"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

CHROMA_HOST = "chroma.tiny-agent.svc.cluster.local"
CHROMA_PORT = 8000


# --------------------------------------------------
# Clients
# --------------------------------------------------

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

chroma = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
)

collection = chroma.get_or_create_collection(
    name="knowledge"
)

manifest_collection = chroma.get_or_create_collection(
    name="document_manifest"
)


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI()


# --------------------------------------------------
# Embeddings
# --------------------------------------------------

def embed(text):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": text,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["embeddings"][0]


# --------------------------------------------------
# MinIO
# --------------------------------------------------

def list_objects():
    objects = []

    for item in minio_client.list_objects(
        BUCKET,
        recursive=True,
    ):
        if not item.is_dir:
            objects.append(item.object_name)

    return objects


def read_object(object_name):
    response = minio_client.get_object(
        BUCKET,
        object_name,
    )

    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def get_etag(object_name):
    stat = minio_client.stat_object(
        BUCKET,
        object_name,
    )

    return stat.etag.strip('"')


# --------------------------------------------------
# Document extraction
# --------------------------------------------------

def extract_document(object_name, data):
    lower = object_name.lower()

    # PDF
    if lower.endswith(".pdf"):
        return extract_pdf(data)

    # TXT
    if lower.endswith(".txt"):
        text = data.decode("utf-8", errors="replace")

        return [
            {
                "text": text,
                "page": None,
            }
        ]

    print(f"Skipping unsupported file: {object_name}")

    return []


def extract_pdf(data):
    pages = []

    document = pymupdf.open(
        stream=data,
        filetype="pdf",
    )

    try:
        for page_number, page in enumerate(document):
            text = page.get_text(
                "text",
                sort=True,
            )

            if text.strip():
                pages.append(
                    {
                        "text": text,
                        "page": page_number + 1,
                    }
                )

    finally:
        document.close()

    return pages


# --------------------------------------------------
# Chunking
# --------------------------------------------------

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def chunk_text(text):
    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + CHUNK_SIZE,
            len(text),
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# --------------------------------------------------
# Stable chunk IDs
# --------------------------------------------------

def chunk_id(object_name, chunk_number):
    value = f"{object_name}:{chunk_number}"

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


# --------------------------------------------------
# Incremental ingestion
# --------------------------------------------------

def get_manifest(object_name):

    result = manifest_collection.get(
        ids=[object_name],
        include=["metadatas"],
    )

    if not result["ids"]:
        return None

    return result["metadatas"][0]


def remove_old_chunks(object_name, old_chunk_count):

    if old_chunk_count <= 0:
        return

    ids = [
        chunk_id(object_name, i)
        for i in range(old_chunk_count)
    ]

    collection.delete(
        ids=ids
    )


def ingest(object_name):

    etag = get_etag(object_name)

    previous = get_manifest(object_name)

    if previous:
        previous_etag = previous.get("etag")

        if previous_etag == etag:
            print(
                f"Skipping unchanged document: {object_name}"
            )
            return False

        old_chunk_count = int(
            previous.get("chunk_count", 0)
        )

        print(
            f"Document changed: {object_name}"
        )

        remove_old_chunks(
            object_name,
            old_chunk_count,
        )

    print(f"Reading: {object_name}")

    data = read_object(object_name)

    print(
        f"Bytes: {len(data)}"
    )

    units = extract_document(
        object_name,
        data,
    )

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    chunk_number = 0

    for unit in units:

        page = unit["page"]

        chunks = chunk_text(unit["text"])

        for chunk in chunks:

            print(
                f"Embedding {object_name} "
                f"chunk {chunk_number}"
            )

            vector = embed(chunk)

            metadata = {
                "source": object_name,
                "chunk": chunk_number,
            }

            if page is not None:
                metadata["page"] = page

            documents.append(chunk)
            embeddings.append(vector)
            metadatas.append(metadata)

            ids.append(
                chunk_id(
                    object_name,
                    chunk_number,
                )
            )

            chunk_number += 1

    if not documents:
        print(
            f"No text found: {object_name}"
        )
        return False

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    manifest_collection.upsert(
        ids=[object_name],
        metadatas=[
            {
                "etag": etag,
                "chunk_count": chunk_number,
            }
        ],
    )

    print(
        f"Ingested {object_name}: "
        f"{chunk_number} chunks"
    )

    return True


# --------------------------------------------------
# Ingest everything
# --------------------------------------------------

def ingest_all():

    objects = list_objects()

    print(
        f"Objects found: {len(objects)}"
    )

    for object_name in objects:

        try:
            ingest(object_name)

        except Exception as e:
            print(
                f"ERROR ingesting "
                f"{object_name}: {e}"
            )


# --------------------------------------------------
# Search
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "collection_count": collection.count(),
    }


@app.get("/search")
def search(
    q: str = Query(...),
    n: int = Query(5),
):

    query_embedding = embed(q)

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=n,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    matches = []

    for i in range(
        len(results["documents"][0])
    ):

        matches.append(
            {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )

    return {
        "query": q,
        "results": matches,
    }


# --------------------------------------------------
# Startup
# --------------------------------------------------

if __name__ == "__main__":

    print("Starting ingestion...")

    ingest_all()

    print(
        f"Collection count: "
        f"{collection.count()}"
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
    )