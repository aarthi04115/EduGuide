import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def ask_llm(question, context):
    prompt = f"""
You are EduGuide, an academic learning assistant.

Answer the student's question using the academic context provided below.

Academic Context:
{context}

Student Question:
{question}

Instructions:
- Use the academic context as the primary source.
- Explain the concept clearly and accurately.
- Do not invent information that is not supported by the context.
- If the context does not contain enough information, say that the available academic material does not provide enough information.
"""

    response = client.chat.completions.create(
        model="openrouter/auto",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content