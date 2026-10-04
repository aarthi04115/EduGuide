
from functools import lru_cache
from pathlib import Path
from threading import Lock

from document_processor import extract_text_from_pdf
from chunker import create_chunks
from embeddings import create_embeddings
from index_cache import create_index_for_chunks
from retriever import search
from llm_service import ask_llm
from college_knowledge import retrieve_college_context
from study_materials import retrieve_context_with_sources


BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "documents" / "BDA 2 marks.pdf"
_academic_index_lock = Lock()


@lru_cache(maxsize=1)
def load_academic_index():
    """Build the academic document index only when first needed."""
    with _academic_index_lock:
        if not PDF_PATH.exists():
            return [], None

        text = extract_text_from_pdf(str(PDF_PATH))
        chunks = create_chunks(text)

        if not chunks:
            return [], None

        index = create_index_for_chunks(chunks, create_embeddings)
        return chunks, index


def answer_question(
    question: str,
    document_ids=None,
    owner_id=None,
    include_sources=False,
    db=None,
):
    college_context, college_sources = retrieve_college_context(
        question, db
    )

    if document_ids:
        if owner_id is None:
            raise ValueError(
                "Document retrieval requires an authenticated owner."
            )

        personal_context, personal_sources = retrieve_context_with_sources(
            question,
            document_ids,
            owner_id,
            db=db,
        )

        answer = ask_llm(
            question,
            personal_context,
            college_context=college_context,
        )
        sources = personal_sources + college_sources

        return (answer, sources) if include_sources else answer

    chunks, index = load_academic_index()

    if index is None or not chunks:
        context = ""
    else:
        query_embedding = create_embeddings([question])

        _, indices = search(
            index,
            query_embedding,
            top_k=min(3, len(chunks)),
        )

        retrieved_chunks = [
            chunks[i]
            for i in indices[0]
            if 0 <= i < len(chunks)
        ]

        context = "\n\n".join(retrieved_chunks)

    answer = ask_llm(
        question,
        context,
        college_context=college_context,
    )

    return (answer, college_sources) if include_sources else answer
