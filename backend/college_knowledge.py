import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from embeddings import create_embeddings
from index_cache import create_index_for_chunks
from models import CollegeChunk, CollegeSource
from retriever import search


MAX_COLLEGE_RETRIEVAL_DISTANCE = 1.3
MAX_COLLEGE_CONTEXT_CHUNKS = 5
MAX_COLLEGE_RETRIEVAL_CHUNKS = 12


@dataclass
class IndexedCollegeChunk:
    content: str
    source_id: str
    source_url: str
    page_title: str
    chunk_index: int


_chunks: list[IndexedCollegeChunk] = []
_index = None
_index_version = None
_index_lock = RLock()


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def rebuild_college_index(db: Session) -> int:
    global _chunks, _index, _index_version
    with _index_lock:
        version = _database_version(db)
        rows = db.execute(
            select(CollegeChunk, CollegeSource)
            .join(CollegeSource, CollegeSource.id == CollegeChunk.source_id)
            .where(CollegeSource.status == "indexed")
            .order_by(CollegeSource.source_url, CollegeChunk.chunk_index)
        ).yield_per(256)
        chunks = [
            IndexedCollegeChunk(
                content=chunk.content,
                source_id=source.id,
                source_url=source.source_url,
                page_title=source.page_title,
                chunk_index=chunk.chunk_index,
            )
            for chunk, source in rows
        ]
        if not chunks:
            _chunks = []
            _index = None
            _index_version = version
            return 0
        index = create_index_for_chunks(
            [chunk.content for chunk in chunks],
            create_embeddings,
        )
        _chunks = chunks
        _index = index
        _index_version = version
        return len(chunks)


def _database_version(db: Session):
    return (
        db.scalar(
            select(func.count(CollegeSource.id)).where(
                CollegeSource.status == "indexed"
            )
        )
        or 0,
        db.scalar(
            select(func.max(CollegeSource.last_indexed_at)).where(
                CollegeSource.status == "indexed"
            )
        ),
        db.scalar(
            select(func.count(CollegeChunk.id))
            .join(CollegeSource)
            .where(CollegeSource.status == "indexed")
        )
        or 0,
    )


def retrieve_college_context(question: str, db: Session | None = None):
    with _index_lock:
        if db is not None and _index_version != _database_version(db):
            rebuild_college_index(db)
        if _index is None or not _chunks:
            return "", []
        query_embedding = create_embeddings([question])
        distances, indices = search(
            _index,
            query_embedding,
            top_k=min(MAX_COLLEGE_RETRIEVAL_CHUNKS, len(_chunks)),
        )
        context_parts = []
        sources = []
        seen_sources = set()
        for distance, chunk_position in zip(distances[0], indices[0]):
            if (
                chunk_position < 0
                or chunk_position >= len(_chunks)
                or float(distance) > MAX_COLLEGE_RETRIEVAL_DISTANCE
            ):
                continue
            chunk = _chunks[chunk_position]
            context_parts.append(
                f"Official Sri Sairam Engineering College source: "
                f"{chunk.page_title}\nURL: {chunk.source_url}\n{chunk.content}"
            )
            if chunk.source_id not in seen_sources:
                sources.append(
                    {
                        "id": chunk.source_id,
                        "filename": chunk.page_title,
                        "title": chunk.page_title,
                        "url": chunk.source_url,
                        "source_type": "college_website",
                    }
                )
                seen_sources.add(chunk.source_id)
            if len(context_parts) >= MAX_COLLEGE_CONTEXT_CHUNKS:
                break
        return "\n\n".join(context_parts), sources


def record_fetch_failure(db: Session, source_url: str, error_code: str):
    source_id = _stable_hash(source_url)
    source = db.get(CollegeSource, source_id)
    now = datetime.now(timezone.utc)
    if source is None:
        source = CollegeSource(
            id=source_id,
            source_url=source_url,
            page_title=source_url,
            content_hash=_stable_hash(""),
            content_text="",
            source_type="html",
            status="failed",
            last_fetched_at=now,
            last_error=error_code[:64],
        )
        db.add(source)
    else:
        source.status = "stale" if source.content_text else "failed"
        source.last_fetched_at = now
        source.last_error = error_code[:64]
    db.commit()


def upsert_college_source(
    db: Session,
    *,
    source_url: str,
    page_title: str,
    content_text: str,
    source_type: str,
    fetched_at: datetime | None = None,
) -> str:
    cleaned_text = content_text.strip()
    if not cleaned_text:
        raise ValueError("Cannot index an empty college source.")
    source_id = _stable_hash(source_url)
    content_hash = _stable_hash(cleaned_text)
    now = fetched_at or datetime.now(timezone.utc)
    source = db.get(CollegeSource, source_id)
    if source is None:
        source = CollegeSource(
            id=source_id,
            source_url=source_url,
            page_title=page_title[:512] or source_url,
            content_hash=content_hash,
            content_text=cleaned_text,
            source_type=source_type,
            status="indexed",
            last_fetched_at=now,
            last_indexed_at=now,
            last_error=None,
        )
        db.add(source)
        db.flush()
    elif source.content_hash == content_hash and source.status == "indexed":
        normalized_title = page_title[:512] or source_url
        if source.page_title != normalized_title:
            source.page_title = normalized_title
            source.last_indexed_at = now
        source.last_fetched_at = now
        source.last_error = None
        db.commit()
        return source.id
    else:
        source.page_title = page_title[:512] or source_url
        source.content_hash = content_hash
        source.content_text = cleaned_text
        source.source_type = source_type
        source.status = "indexed"
        source.duplicate_of_id = None
        source.last_fetched_at = now
        source.last_indexed_at = now
        source.last_error = None
        db.query(CollegeChunk).filter(
            CollegeChunk.source_id == source.id
        ).delete(synchronize_session=False)

    duplicate = db.scalar(
        select(CollegeSource).where(
            CollegeSource.content_hash == content_hash,
            CollegeSource.id != source_id,
            CollegeSource.status == "indexed",
        )
    )
    if duplicate is not None:
        source.status = "duplicate"
        source.duplicate_of_id = duplicate.id
        db.commit()
        return source.id

    from chunker import create_chunks

    chunks = create_chunks(cleaned_text)
    if not chunks:
        raise ValueError("No text chunks were produced for the college source.")
    for chunk_index, content in enumerate(chunks):
        db.add(
            CollegeChunk(
                id=_stable_hash(f"{source_id}:{chunk_index}"),
                source_id=source_id,
                chunk_index=chunk_index,
                content=content,
            )
        )
    db.commit()
    return source.id


def college_source_statistics(db: Session):
    sources = db.scalars(
        select(CollegeSource).order_by(CollegeSource.source_url)
    ).all()
    counts = {
        "indexed": 0,
        "stale": 0,
        "failed": 0,
        "duplicate": 0,
        "chunks": db.query(CollegeChunk).count(),
    }
    for source in sources:
        counts[source.status] = counts.get(source.status, 0) + 1
    return counts
