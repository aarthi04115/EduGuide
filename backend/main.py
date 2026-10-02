from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import answer_question

app = FastAPI(title = "EduGuide API")

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "EduGuide backend is running"}

@app.post("/chat")
def chat(request: ChatRequest):

    print("QUESTION RECEIVED:", request.question)

    answer = answer_question(request.question)

    print("ANSWER GENERATED:", answer)

    return {
        "question": request.question,
        "answer": answer
    }