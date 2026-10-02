from document_processor import extract_text_from_pdf
from chunker import create_chunks
from embeddings import create_embeddings
from retriever import create_index, search
from llm_service import ask_llm


PDF_PATH = "data/documents/BDA 2 marks.pdf"


# ---------- Load academic document ----------
text = extract_text_from_pdf(PDF_PATH)

chunks = create_chunks(text)

embeddings = create_embeddings(chunks)

index = create_index(embeddings)


# ---------- RAG function ----------
def answer_question(question: str):

    # 1. Convert question into embedding
    query_embedding = create_embeddings([question])

    # 2. Search FAISS
    distances, indices = search(
        index,
        query_embedding,
        top_k=3
    )

    # 3. Get the actual text of retrieved chunks
    retrieved_chunks = []

    for i in range(3):
        chunk_index = indices[0][i]
        retrieved_chunks.append(chunks[chunk_index])

    # 4. Combine retrieved chunks into context
    context = "\n\n".join(retrieved_chunks)

    # 5. Send question + context to LLM
    answer = ask_llm(question, context)

    return answer