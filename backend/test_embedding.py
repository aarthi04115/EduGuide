from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "What is DBMS?"

embedding = model.encode(text)

print("Number of values:", len(embedding))
print("First 10 values:", embedding[:10])