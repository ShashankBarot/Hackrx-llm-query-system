# 📄 main.py — HackRx LLM Query System (with Groq + LLaMA 3)

import os
import numpy as np
import requests
import json

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

# === Dummy data for now (will replace with actual chunks later) ===
document_chunks = [
    "Knee surgery is covered for insured persons under the age of 60.",
    "A grace period of 30 days is allowed for premium payment.",
    "There is a 2-year waiting period for cataract surgery.",
    "Maternity expenses are covered after 24 months of continuous policy.",
    "No Claim Discount of 5% is applicable if no claims are made."
]

# === Embed and index chunks ===
embeddings = model.encode(document_chunks)
dim = len(embeddings[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings).astype('float32'))

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
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ LLaMA 3 Error: {e}"

# === Main API Endpoint ===
@app.post("/hackrx/run")
def run_query(input: QueryInput):
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
