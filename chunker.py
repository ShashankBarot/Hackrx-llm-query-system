# chunker.py (in-memory, no file saves)

import fitz  # PyMuPDF
import requests
from io import BytesIO

# Step 1: Get text from a blob URL (without saving)
def extract_text_from_blob_url(blob_url):
    try:
        response = requests.get(blob_url)
        response.raise_for_status()
        pdf_bytes = BytesIO(response.content)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
        return text
    except Exception as e:
        print("❌ PDF text extraction error:", e)
        return ""

# Step 2: Chunk text
def chunk_text(text, max_words=600):
    words = text.split()
    chunks = []
    chunk = []

    for word in words:
        chunk.append(word)
        if len(chunk) >= max_words:
            chunks.append(" ".join(chunk))
            chunk = []

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks

# Combined function
def process_pdf_from_url(blob_url, chunk_size=300):
    text = extract_text_from_blob_url(blob_url)
    if not text:
        return []
    return chunk_text(text, max_words=chunk_size)
