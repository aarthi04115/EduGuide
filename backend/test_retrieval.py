from document_processor import extract_text_from_pdf
from chunker import create_chunks
from embeddings import create_embeddings
from retriever import create_index, search



pdf_path = "data/documents/BDA 2 marks.pdf"

# 1. Extract text
text = extract_text_from_pdf(pdf_path)

# 2. Create chunks
chunks = create_chunks(text)

# 3. Create embeddings for chunks
embeddings = create_embeddings(chunks)

# 4. Create FAISS index
index = create_index(embeddings)

# 5. Student question
question = "What is Volume in Big Data?"

# 6. Convert question into embedding
query_embedding = create_embeddings([question])

# 7. Search FAISS
distances, indices = search(index, query_embedding, top_k=3)

# 8. Display results
for i in range(3):
    chunk_index = indices[0][i]

    print("\n--- RESULT", i + 1, "---")
    print("Chunk index:", chunk_index)
    print("Distance:", distances[0][i])
    print(chunks[chunk_index])