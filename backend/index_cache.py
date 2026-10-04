import hashlib
import logging
import os
import uuid
from pathlib import Path

import faiss

from retriever import add_embeddings, create_index


logger = logging.getLogger(__name__)
MODEL_NAME = "all-MiniLM-L6-v2-float16"
EMBEDDING_BATCH_SIZE = 32
MAX_CACHED_INDEX_FILES = 64
BASE_DIR = Path(__file__).resolve().parent
INDEX_CACHE_DIR = Path(
    os.getenv("EDUGUIDE_INDEX_CACHE_DIR", str(BASE_DIR / "data" / "index_cache"))
).resolve()


def _cache_key(chunks):
    digest = hashlib.sha256(MODEL_NAME.encode("ascii"))
    for chunk in chunks:
        encoded = chunk.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _cache_path(key):
    return INDEX_CACHE_DIR / f"{key}.faiss"


def _prune_cache(exclude):
    cache_files = sorted(
        (
            path
            for path in INDEX_CACHE_DIR.glob("*.faiss")
            if path != exclude
        ),
        key=lambda path: path.stat().st_mtime,
    )
    excess = len(cache_files) + 1 - MAX_CACHED_INDEX_FILES
    for path in cache_files[: max(excess, 0)]:
        try:
            path.unlink()
        except OSError:
            logger.warning("Could not remove an expired FAISS cache file.")


def _save_index(path, index):
    temporary_path = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp")
    try:
        faiss.write_index(index, str(temporary_path))
        temporary_path.replace(path)
        _prune_cache(path)
    except (OSError, RuntimeError):
        logger.exception("Could not persist a reusable FAISS index.")
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not clean up a temporary FAISS index file.")


def create_index_for_chunks(chunks, embedder):
    if not chunks:
        raise ValueError("Cannot create a FAISS index without text chunks.")

    cache_path = _cache_path(_cache_key(chunks))
    if cache_path.is_file():
        try:
            index = faiss.read_index(str(cache_path))
            if index.ntotal == len(chunks):
                os.utime(cache_path, None)
                return index
            logger.warning("Ignoring a FAISS cache with an unexpected vector count.")
        except (OSError, RuntimeError):
            logger.warning("Ignoring an unreadable FAISS cache file.")

    index = None
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        embeddings = embedder(chunks[start : start + EMBEDDING_BATCH_SIZE])
        if index is None:
            index = create_index(embeddings)
        else:
            add_embeddings(index, embeddings)

    if index is None or index.ntotal != len(chunks):
        raise RuntimeError("The FAISS index does not contain every text chunk.")

    try:
        INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("Could not create the FAISS index cache directory.")
        return index

    _save_index(cache_path, index)
    return index
