import chromadb
from chromadb.config import Settings

# Initialize Chroma client (local, persistent)
client = chromadb.Client(
    Settings(
        persist_directory="vector_store",
        anonymized_telemetry=False
    )
)

# Create or get collection
collection = client.get_or_create_collection(
    name="knowledge_chunks"
)

def add_chunk_to_vector_store(chunks, document_id):
    for chunk in chunks:
        collection.add(
            ids=[f"{document_id}_{chunk['chunk_id']}"], # unique id for each chunk
            document = [chunk["text"]],
            metadatas = [{
                "document_id": document_id,
                "chunk_id": chunk["chunk_id"],
                "start_word": chunk["start_word"],
                "end_word": chunk["end_word"]
            }]
        )
        
        
def query_vector_store(query_embedding, top_k=3):
    results = collection.query(
        query_embedding=[query_embedding],
        n_results=top_k
    )
    return results