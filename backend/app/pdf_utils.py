import pdfplumber
import re

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "/n"
                
                
    return text

def clean_text(raw_text: str) -> str:
    text = raw_text
    
    # Replace multiple whitespace characters with a single space
    text = re.sub(r"\s+", " ", text) 
    
    # Remove non breaking spaces
    text = text.replace("\u00a0", " ")
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text

def extract_and_clean_text(file_path: str) -> str:
    raw_text = extract_text_from_pdf(file_path)
    clean = clean_text(raw_text)
    return clean