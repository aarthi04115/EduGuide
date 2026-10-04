import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Add your Groq API key to the project .env file."
    )

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)



def ask_llm(question, context):
    system_prompt = """
You are EduGuide, an intelligent academic doubt-support assistant
for students.

Your goal is to help students understand academic concepts clearly.

Follow these rules:
1. First, check whether the academic context is relevant to the question.
2. If the context contains a relevant answer, use it as the primary
   source and explain it clearly.
3. If the context is empty, irrelevant, or insufficient, answer using
   your general academic knowledge.
4. When using general knowledge because the provided material is
   insufficient, briefly mention that the answer supplements the
   available study material.
5. Never claim that information comes from the student's notes unless
   it is actually supported by the provided context.
6. If you are uncertain about a fact, say so instead of inventing it.
7. Use simple language, examples, and step-by-step explanations
   when helpful.
8. Stay focused on education and academic learning.
"""

    user_prompt = f"""
Academic context retrieved from study materials:
{context if context and str(context).strip() else "No relevant academic material was retrieved."}

Student question:
{question}

Answer the student's question following your instructions.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content
