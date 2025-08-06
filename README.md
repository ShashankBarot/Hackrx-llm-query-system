# 🚀 HackRx LLM Query-Retrieval System 🤖

This is an LLM-powered intelligent query-retrieval system for HackRx. It extracts content from PDF documents (via URL), chunks the content into manageable pieces, and uses an LLM (Groq's LLaMA 3) to answer natural language questions.

---

## 🛠️ Features

- 📥 Fetches PDF from a given URL
- 🧾 Extracts text using PyMuPDF (fitz)
- 🧠 Splits text into readable chunks
- 🔎 Embeds chunks using `sentence-transformers`
- 📡 Retrieves top relevant chunks using FAISS
- 💬 Sends those chunks to LLaMA 3 (Groq API) to generate a justified response

---

## 📦 Tech Stack

- FastAPI
- PyMuPDF (`fitz`)
- FAISS (in-memory)
- Sentence-Transformers (`all-MiniLM-L6-v2`)
- Groq API (LLaMA 3)
- Uvicorn

---

## 🧪 How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Hackrx-llm-query-system.git

cd Hackrx-llm-query-system
```

### 2.  Install Dependencies

```bash
pip install -r requirements.txt
```

### 3.   Set Up Environment Variables
Create a .env file:

```bash
GROQ_API_KEY=your_actual_groq_key_here
GROQ_MODEL=llama3-70b-8192
```

### 4. Run the API
```bash
uvicorn main:app --reload
```
The app will be accessible at:
```bash
http://localhost:8000
```
### 5. Test the API
```bash
POST /hackrx/run
```
With a JSON body like:

```bash
{
  "documents": "https://example.com/your-policy.pdf",
  "questions": [
    "What is the waiting period for pre-existing diseases?",
    "Is dental treatment covered?"
  ]
}
```

---
### ✅ Sample curl command
```bash
curl -X POST http://localhost:8000/hackrx/run \
  -H "Content-Type: application/json" \
  -d '{"documents":"https://example.com/your-policy.pdf","questions":["What is covered under AYUSH treatment?"]}'
```
---
## ⚠️ Hosting Note

This app is deployed on **Replit**. If it sleeps due to inactivity:
- Visit the link once in your browser to wake it up
- Then retry your POST request
---
###  Built By : Bit_By_Bit