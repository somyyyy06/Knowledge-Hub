def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
):
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "start_word": start,
            "end_word": end
        })
        
        chunk_id += 1
        start = end - overlap
        
        if start < 0:
            start = 0
            
    return chunks