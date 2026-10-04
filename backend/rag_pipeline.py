from document_processor import extract_text_from_pdf
from chunker import create_chunks
from embeddings import create_embeddings
from retriever import create_index, search
from llm_service import ask_llm
from college_knowledge import retrieve_college_context
from study_materials import retrieve_context_with_sources


PDF_PATH = "data/documents/BDA 2 marks.pdf"


# ---------- Load academic document ----------
text = extract_text_from_pdf(PDF_PATH)

chunks = create_chunks(text)

embeddings = create_embeddings(chunks)

index = create_index(embeddings)


# ---------- RAG function ----------
def answer_question(
    question: str,
    document_ids=None,
    owner_id=None,
    include_sources=False,
    db=None,
):
    college_context, college_sources = retrieve_college_context(question, db)
    if document_ids:
        if owner_id is None:
            raise ValueError("Document retrieval requires an authenticated owner.")
        personal_context, personal_sources = retrieve_context_with_sources(
            question,
            document_ids,
            owner_id,
        )
        answer = ask_llm(
            question,
            personal_context,
            college_context=college_context,
        )
        sources = personal_sources + college_sources
        return (answer, sources) if include_sources else answer

    # 1. Convert question into embedding
    query_embedding = create_embeddings([question])

    # 2. Search FAISS
    _, indices = search(
        index,
        query_embedding,
        top_k=min(3, len(chunks))
    )

    # 3. Get the actual text of retrieved chunks
    retrieved_chunks = []

    for chunk_index in indices[0]:
        if 0 <= chunk_index < len(chunks):
            retrieved_chunks.append(chunks[chunk_index])

    # 4. Combine retrieved chunks into context
    context = "\n\n".join(retrieved_chunks)

    # 5. Send question + context to LLM
    answer = ask_llm(question, context, college_context=college_context)

    return (answer, college_sources) if include_sources else answer