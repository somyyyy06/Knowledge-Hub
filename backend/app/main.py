from fastapi import FastAPI, UploadFile, File, HTTPException
from app.pdf_utils import extract_text_from_pdf
from app.chunking import chunk_text
from app.embeddings import generate_embeddings
from app.vector_store import add_chunk_to_vector_store, query_vector_store
import os
import uuid

app = FastAPI()

UPLOAD_DIR = "uploads/documents"

os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload") 
async def upload_file(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    #  Used to generate a unique filename for the uploaded file to avoid conflicts     
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    # Save file in chunks
    with open(file_path, "wb") as buffer:
        while True:
            chunk = await file.read(1024) # Read the file in chunks to handle large files
            if not chunk:
                break
            buffer.write(chunk)
        
    # Reset file pointer (important for later processing)
    await file.seek(0)
    
    text = extract_text_from_pdf(file_path)
    print("\n--- EXTRACTED TEXT SAMPLE ---\n")
    print(text[:1000])
    print("\n--- END SAMPLE ---\n")
    
    chunks = chunk_text(text)
    print(f"Total chunks created: {len(chunks)}")
    
    add_chunk_to_vector_store(chunks, document_id="doc1")
    
    question = "What is the transformer architecture?"
    question_embedding = generate_embedding(question)
    results = query_vector_store(question_embedding)
    
    print("\n--- RETRIEVED CHUNKS ---\n")
    for doc in results["documents"][0]:
        print(doc[:300])
        print("-----")
        
    return {
        "original_filename": file.filename,
        "stored_as": unique_filename,
        "content_type": file.content_type,
        "message": "File uploaded successfully"
    }



@app.get("/health")
def health_check():
    return {"status": "ok"}