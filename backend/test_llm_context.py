from llm_service import ask_llm

question = "What is Volume in Big Data?"

context = """
Volume is one of the major characteristics of Big Data.
It refers to the enormous size or amount of data.
Examples include massive digital archives of scanned medical records
and customer correspondence.
"""

answer = ask_llm(question, context)

print("\n--- ANSWER ---")
print(answer)