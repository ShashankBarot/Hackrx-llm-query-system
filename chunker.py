# chunker.py - In-memory PDF processing without file saves

import fitz  # PyMuPDF
import requests
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_blob_url(blob_url):
    """Extract text from PDF URL without saving to disk"""
    try:
        logger.info(f"Downloading PDF from: {blob_url}")
        response = requests.get(blob_url, timeout=30)
        response.raise_for_status()
        
        # Process PDF in memory
        pdf_bytes = BytesIO(response.content)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        text_pages = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_pages.append(page.get_text())
        
        doc.close()
        full_text = "\n".join(text_pages)
        
        logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
        return full_text
        
    except requests.RequestException as e:
        logger.error(f"Error downloading PDF: {e}")
        return ""
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

def chunk_text(text, max_words=300):
    """Split text into chunks of specified word count"""
    if not text.strip():
        return []
    
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # Add remaining words as final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logger.info(f"Text split into {len(chunks)} chunks")
    return chunks

def process_pdf_from_url(blob_url, chunk_size=300):
    """Complete pipeline: download PDF, extract text, and create chunks"""
    text = extract_text_from_blob_url(blob_url)
    if not text:
        logger.warning("No text extracted from PDF")
        return []
    
    return chunk_text(text, max_words=chunk_size)