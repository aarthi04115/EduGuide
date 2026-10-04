import faiss
import numpy as np


def create_index(embeddings):
    vectors = np.asarray(embeddings, dtype="float32").copy()
    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    return index


def add_embeddings(index, embeddings):
    vectors = np.asarray(embeddings, dtype="float32").copy()
    faiss.normalize_L2(vectors)
    index.add(vectors)


def search(index, query_embedding, top_k=3):
    queries = np.asarray(query_embedding, dtype="float32").copy()
    faiss.normalize_L2(queries)

    distances, indices = index.search(
        queries,
        min(top_k, index.ntotal)
    )

    return distances, indices