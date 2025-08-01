# 📄 main.py — HackRx LLM Query System (with Groq + LLaMA 3)

import os
import numpy as np
import requests
import json

from chunker import process_pdf_and_chunk
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

# Load .env
load_dotenv()

# === Load Groq API credentials ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

# === Load embedding model ===
model = SentenceTransformer('all-MiniLM-L6-v2')


# === FastAPI app ===
app = FastAPI()
# === Request model ===
class QueryInput(BaseModel):
    documents: str  # not used in dummy mode
    questions: list[str]

# === Call Groq LLaMA 3 ===
def get_llama3_answer(question, context):
    prompt = f"""Answer the following question using only the context provided.

Question: {question}

Context:
{context}

Answer:"""

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant for insurance policy understanding."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 200
            }
        )
        result = response.json()

        if "choices" not in result or len(result["choices"]) == 0:
            return "❌ LLaMA 3 Error: No response received."

        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"❌ LLaMA 3 Error: {e}"

# === Main API Endpoint ===
@app.post("/hackrx/run")

def run_query(input: QueryInput):
    # 🔄 Step 1: Get chunks from the PDF blob URL
    document_chunks = process_pdf_and_chunk(input.documents)

    if not document_chunks:
        return {"error": "Failed to process PDF or no text extracted."}

    # 🔄 Step 2: Embed all chunks
    embeddings = model.encode(document_chunks)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))

    results = []
    for question in input.questions:
        query_embedding = model.encode([question])
        D, I = index.search(np.array(query_embedding).astype("float32"), k=3)
        matched = [document_chunks[i] for i in I[0]]

        context = "\n".join(matched)
        answer = get_llama3_answer(question, context)

        results.append({
            "answer": answer,
            "justification": "Based on these document clauses:\n- " + "\n- ".join(matched)
        })

    return {"answers": results}