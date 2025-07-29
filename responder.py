def generate_answer(question, relevant_chunks):
    context = "\n".join(relevant_chunks)
    return f"Question: {question}\n\nBased on the context:\n{context}"
