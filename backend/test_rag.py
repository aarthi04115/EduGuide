from document_processor import extract_text_from_pdf
from chunker import create_chunks
from embeddings import create_embeddings


pdf_path = "data/documents/BDA 2 marks.pdf"

text = extract_text_from_pdf(pdf_path)

chunks = create_chunks(text)

embeddings = create_embeddings(chunks)

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)