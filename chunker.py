# chunker.py

import fitz  # PyMuPDF
import requests
import json
import os

# Step 1: Download the PDF from a blob URL
def download_pdf(blob_url, save_path="document.pdf"):
    try:
        response = requests.get(blob_url)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        print("✅ PDF downloaded successfully.")
        return save_path
    except Exception as e:
        print("❌ Error downloading PDF:", e)
        return None

# Step 2: Extract text from PDF
def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text() for page in doc])
        print("✅ Text extracted from PDF.")
        return text
    except Exception as e:
        print("❌ Error extracting text:", e)
        return ""

# Step 3: Chunk text into smaller parts
def chunk_text(text, max_words=300):
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

    print(f"✅ Text split into {len(chunks)} chunks.")
    return chunks

# Optional: Save chunks to a JSON file
def save_chunks_to_json(chunks, filename="chunks.json"):
    try:
        with open(filename, "w") as f:
            json.dump(chunks, f, indent=2)
        print(f"✅ Chunks saved to {filename}")
    except Exception as e:
        print("❌ Error saving chunks to JSON:", e)

# Combined function to run full chunking pipeline
def process_pdf_and_chunk(blob_url, chunk_size=300):
    file_path = download_pdf(blob_url)
    if not file_path:
        return []

    text = extract_text_from_pdf(file_path)
    if not text:
        return []

    chunks = chunk_text(text, max_words=chunk_size)
    return chunks

if __name__ == "__main__":
    blob_url = input("🔗 Enter the Blob URL of the PDF: ").strip()
    chunk_size_input = input("🔢 Enter chunk size (default 300): ").strip()
    save_option = input("💾 Do you want to save chunks to JSON? (y/n): ").strip().lower()

    chunk_size = int(chunk_size_input) if chunk_size_input.isdigit() else 300
    chunks = process_pdf_and_chunk(blob_url, chunk_size=chunk_size)

    if chunks:
        print("\n📌 First Chunk Preview:\n")
        print(chunks[0][:500])

        if save_option == 'y':
            save_chunks_to_json(chunks)
        else:
            print("📝 Chunks not saved.")
