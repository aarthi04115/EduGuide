import fitz
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page in document:
        text += page.get_text()
    document.close()
    return extract_text_from_pdf

pdf_path = "data/documents/BDA 2 marks.pdf"

from chunker import create_chunks
text = extract_text_from_pdf(pdf_path)
chunks = create_chunks(text)
print("Total characters:", len(text))
print("Total chunks:", len(chunks))
print("\nFIRST CHUNK:")
print(chunks[0])