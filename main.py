# main.py — HackRx LLM Query System with Groq + LLaMA 3

import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from chunker import process_pdf_from_url

load_dotenv()

# Load Groq API credentials
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

app = FastAPI(
    title="HackRx LLM Query System",
    description="AI-powered PDF document analysis with Groq LLM integration",
    version="1.0.0"
)

# Simple keyword-based fallback search
def simple_similarity_search(question, chunks, top_k=3):
    question_words = set(question.lower().split())
    scores = []
    for i, chunk in enumerate(chunks):
        chunk_words = set(chunk.lower().split())
        overlap = len(question_words.intersection(chunk_words))
        scores.append((overlap, i))
    scores.sort(reverse=True)
    return [chunks[i] for _, i in scores[:top_k]]

# Request format
class QueryInput(BaseModel):
    documents: str
    questions: list[str]

def get_llama3_answer(question, context):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    
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
                    {"role": "system", "content": "You are a helpful assistant for document analysis and understanding."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            }
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"

@app.post("/hackrx/run")
def run_query(input: QueryInput):
    chunks = process_pdf_from_url(input.documents)
    if not chunks:
        raise HTTPException(status_code=400, detail="Failed to process document or no content found")

    answers = []
    for question in input.questions:
        matched = simple_similarity_search(question, chunks, top_k=3)
        context = "\n".join(matched)
        answer = get_llama3_answer(question, context)
        answers.append({
            "answer": answer,
            "justification": "Based on these document chunks:\n- " + "\n- ".join(matched)
        })

    return {"answers": answers}

@app.get("/")
def root():
    return {
        "message": "HackRx LLM Query System",
        "version": "1.0.0",
        "endpoint": "/hackrx/run"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "HackRx LLM Query System"}

# Handle HEAD requests for uptime monitoring
@app.head("/health")
def health_check_head():
    return {}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
