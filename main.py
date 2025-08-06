# main.py — stateless, lightweight version

import os
import numpy as np
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
from chunker import process_pdf_from_url

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
app = FastAPI()

class QueryInput(BaseModel):
    documents: str  # Blob URL
    questions: list[str]

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

@app.post("/hackrx/run")
def run_query(input: QueryInput):
    chunks = process_pdf_from_url(input.documents)
    if not chunks:
        return {"error": "❌ Failed to process document or no content found."}

    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype("float32"))

    results = []
    for question in input.questions:
        q_embedding = model.encode([question])
        D, I = index.search(np.array(q_embedding).astype("float32"), k=3)
        matched = [chunks[i] for i in I[0]]

        context = "\n".join(matched)
        answer = get_llama3_answer(question, context)
        results.append({
            "answer": answer,
            "justification": "Based on these document chunks:\n- " + "\n- ".join(matched)
        })

    return {"answers": results}
