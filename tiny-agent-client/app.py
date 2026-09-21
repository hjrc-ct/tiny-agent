import requests
import chromadb


OLLAMA_URL = "http://ollama.tiny-agent.svc.cluster.local:11434"
CHROMA_HOST = "chroma.tiny-agent.svc.cluster.local"
CHROMA_PORT = 8000

MODEL = "qwen3-embedding:0.6b"


# --------------------------------------------------
# Ollama
# --------------------------------------------------

def embed(text):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": MODEL,
            "input": text,
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    return data["embeddings"][0]


# --------------------------------------------------
# Chroma
# --------------------------------------------------

chroma = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
)

collection = chroma.get_or_create_collection(
    name="knowledge"
)


# --------------------------------------------------
# Test document
# --------------------------------------------------

document = """
Camunda 8 uses Zeebe as its workflow engine.

Zeebe executes BPMN processes using a distributed architecture.
A Camunda 8 cluster can contain multiple Zeebe brokers.

Camunda 8 also provides Operate for monitoring processes,
Tasklist for human tasks, and Connectors for integrating
external systems.
"""


# --------------------------------------------------
# Simple chunking
# --------------------------------------------------

chunks = [
    chunk.strip()
    for chunk in document.split("\n\n")
    if chunk.strip()
]


# --------------------------------------------------
# Generate embeddings
# --------------------------------------------------

embeddings = []

for chunk in chunks:
    print("Embedding:", chunk[:80])

    vector = embed(chunk)

    print("Vector dimensions:", len(vector))

    embeddings.append(vector)


# --------------------------------------------------
# Store in Chroma
# --------------------------------------------------

ids = [
    f"camunda-test-{i}"
    for i in range(len(chunks))
]

metadatas = [
    {
        "source": "camunda-test.txt",
        "chunk": i,
    }
    for i in range(len(chunks))
]


collection.upsert(
    ids=ids,
    documents=chunks,
    embeddings=embeddings,
    metadatas=metadatas,
)


print()
print("Stored", len(chunks), "chunks in Chroma.")
print("Collection count:", collection.count())