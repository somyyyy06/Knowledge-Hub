import chromadb
from chromadb.config import Settings
from app.embeddings import generate_embeddings

# Persistent storage on disk
chroma_client = chromadb.Client(
    Settings(
        persist_directory="chroma_db",
        anonymized_telemetry=False
    )
)

collection = chroma_client.get_or_create_collection(
    name="documents"
)


def add_chunk_to_vector_store(chunks, document_id: str):
    """
    Stores chunks with metadata so multiple documents can coexist
    """

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    for chunk in chunks:
        text = chunk["text"]
        chunk_id = chunk["chunk_id"]

        documents.append(text)
        embeddings.append(generate_embeddings(text))
        metadatas.append({
            "document_id": document_id,
            "chunk_id": chunk_id
        })
        ids.append(f"{document_id}_{chunk_id}")

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    chroma_client.persist()


def query_vector_store(query_embedding, top_k=5, document_id=None):
    where_filter = None
    if document_id:
        where_filter = {"document_id": document_id}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "distances", "metadatas"]
    )

    return results

