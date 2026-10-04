import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from chunker import create_chunks
from document_processor import extract_pages_from_file
from embeddings import create_embeddings
from models import Document
from retriever import create_index, search


logger = logging.getLogger(__name__)


@dataclass
class StudyMaterial:
    document_id: str
    owner_id: str
    filename: str
    file_path: str
    size: int
    chunks: list[str]
    page_numbers: list[int]
    index: object

    def as_dict(self, status="indexed", created_at=None):
        result = {
            "id": self.document_id,
            "filename": self.filename,
            "size": self.size,
            "status": status,
        }
        if created_at is not None:
            result["created_at"] = created_at.isoformat()
        return result


_materials: dict[str, StudyMaterial] = {}
MAX_RETRIEVAL_DISTANCE = 1.3


def _chunks_with_pages(pages):
    chunks = []
    page_numbers = []
    for page_number, text in pages:
        page_chunks = create_chunks(text)
        chunks.extend(page_chunks)
        page_numbers.extend([page_number] * len(page_chunks))
    return chunks, page_numbers


def add_material(document_id, owner_id, filename, file_path, pages, size):
    chunks, page_numbers = _chunks_with_pages(pages)
    if not chunks:
        raise ValueError("The document does not contain readable text.")

    index = create_index(create_embeddings(chunks))
    material = StudyMaterial(
        document_id=document_id,
        owner_id=owner_id,
        filename=filename,
        file_path=str(Path(file_path).resolve()),
        size=size,
        chunks=chunks,
        page_numbers=page_numbers,
        index=index,
    )
    _materials[document_id] = material
    return material


def initialize_materials(db: Session):
    for document in db.scalars(
        select(Document).where(Document.status == "indexed")
    ):
        if document.id in _materials:
            continue
        try:
            pages = extract_pages_from_file(document.storage_path)
            if not any(text.strip() for _, text in pages):
                raise ValueError("No readable text was found.")
            chunks, page_numbers = _chunks_with_pages(pages)
            if not chunks:
                raise ValueError("No text chunks were produced.")
            _materials[document.id] = StudyMaterial(
                document_id=document.id,
                owner_id=document.user_id,
                filename=document.filename,
                file_path=document.storage_path,
                size=document.file_size,
                chunks=chunks,
                page_numbers=page_numbers,
                index=create_index(create_embeddings(chunks)),
            )
        except (OSError, ValueError):
            logger.warning(
                "Could not restore indexed study material %s.",
                document.id,
            )
        except Exception:
            logger.exception(
                "Failed to restore indexed study material %s.",
                document.id,
            )


def get_material(document_id, owner_id):
    material = _materials.get(document_id)
    if material is None or material.owner_id != owner_id:
        return None
    return material


def list_materials(owner_id, db: Session):
    documents = db.scalars(
        select(Document)
        .where(Document.user_id == owner_id)
        .order_by(Document.created_at.desc())
    )
    return [
        {
            "id": document.id,
            "filename": document.filename,
            "size": document.file_size,
            "status": (
                "failed"
                if document.status == "indexed"
                and get_material(document.id, owner_id) is None
                else document.status
            ),
            "created_at": document.created_at.isoformat(),
        }
        for document in documents
    ]


def remove_material(document_id):
    return _materials.pop(document_id, None)


def retrieve_context(question, document_ids, owner_id):
    context, _ = retrieve_context_with_sources(question, document_ids, owner_id)
    return context


def retrieve_context_with_sources(question, document_ids, owner_id):
    materials = [
        get_material(document_id, owner_id)
        for document_id in document_ids
    ]
    if any(material is None for material in materials):
        raise ValueError("A selected study material is not available.")

    query_embedding = create_embeddings([question])
    context_parts = []
    sources = []
    for material in materials:
        top_k = min(3, len(material.chunks))
        distances, indices = search(
            material.index,
            query_embedding,
            top_k=top_k,
        )
        for distance, chunk_index in zip(distances[0], indices[0]):
            if (
                0 <= chunk_index < len(material.chunks)
                and float(distance) <= MAX_RETRIEVAL_DISTANCE
            ):
                page_number = material.page_numbers[chunk_index]
                context_parts.append(
                    f"Study material: {material.filename}, page {page_number}\n"
                    f"{material.chunks[chunk_index]}"
                )
                source = {
                    "id": material.document_id,
                    "filename": material.filename,
                    "page_number": page_number,
                }
                if source not in sources:
                    sources.append(source)
    return "\n\n".join(context_parts), sources
