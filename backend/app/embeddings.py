from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(text: str):
    embedding = model.encode(text)  # Converts text → vector (list of floats)
    return embedding

