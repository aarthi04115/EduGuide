# 🎓 EduGuide – Academic Learning Assistant

EduGuide is an **LLM-powered academic learning assistant** that uses **Retrieval-Augmented Generation (RAG)** to answer student questions using pre-loaded academic learning materials.

Instead of relying only on the LLM's general knowledge, EduGuide retrieves relevant content from academic documents and provides that content to the LLM as context before generating an answer.

---

## 🚀 Current Features

* 📚 Academic PDF-based knowledge retrieval
* 🔎 Semantic search using vector embeddings
* 🧠 FAISS-based similarity search
* 🤖 LLM-powered answer generation
* 🔗 Retrieval-Augmented Generation (RAG)
* ⚡ FastAPI backend
* 📖 Context-grounded academic responses
* 🔐 Environment-based API key management

---

## 🏗️ System Architecture

```text
                    Student
                       │
                       ▼
                Chat / API Request
                       │
                       ▼
                  FastAPI
                       │
                       ▼
                RAG Pipeline
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
       Query Embedding      Academic PDF
             │                   │
             ▼                   ▼
          FAISS            Text Extraction
             │                   │
             ▼                   ▼
     Relevant Chunks          Chunking
             │
             └─────────┬─────────┘
                       │
                       ▼
                  Context
                       │
                       ▼
                     LLM
                       │
                       ▼
                  Answer
                       │
                       ▼
                   Student
```

---

## 🔄 How RAG Works in EduGuide

EduGuide follows these steps when a student asks a question:

### 1. Student Question

Example:

```text
What is Big Data?
```

### 2. Query Embedding

The question is converted into a numerical vector using:

```text
all-MiniLM-L6-v2
```

The resulting embedding contains **384 numerical values**.

### 3. Semantic Retrieval

FAISS compares the question embedding with the embeddings of the academic document chunks.

The most relevant chunks are retrieved based on vector similarity.

### 4. Context Construction

The retrieved academic chunks are combined into a context.

```text
Student Question
       +
Retrieved Academic Content
       ↓
     Context
```

### 5. LLM Generation

The question and retrieved context are sent to the LLM.

The LLM generates an answer based primarily on the provided academic material.

---

## 🧩 Technology Stack

| Component            | Technology            |
| -------------------- | --------------------- |
| Programming Language | Python                |
| Backend Framework    | FastAPI               |
| API Server           | Uvicorn               |
| PDF Processing       | PyMuPDF               |
| Text Embeddings      | Sentence Transformers |
| Embedding Model      | all-MiniLM-L6-v2      |
| Vector Search        | FAISS                 |
| LLM Access           | OpenRouter            |
| LLM Client           | OpenAI Python SDK     |
| API Documentation    | Swagger / OpenAPI     |
| Version Control      | Git + GitHub          |

---

## 📁 Project Structure

```text
EduGuide/
│
├── .env
├── .gitignore
├── README.md
│
└── backend/
    │
    ├── main.py
    ├── llm_service.py
    ├── rag_pipeline.py
    ├── document_processor.py
    ├── chunker.py
    ├── embeddings.py
    ├── retriever.py
    │
    ├── data/
    │   └── documents/
    │       └── BDA 2 marks.pdf
    │
    └── test files
        ├── test_embedding.py
        ├── test_llm_context.py
        ├── test_rag.py
        ├── test_rag_pipeline.py
        └── test_retrieval.py
```

---

## ⚙️ Core Components

### `document_processor.py`

Extracts text from academic PDF documents using PyMuPDF.

```text
PDF
 ↓
Extracted Text
```

### `chunker.py`

Splits the extracted document into smaller overlapping chunks.

Current configuration:

```text
Chunk Size: 1000 characters
Overlap: 200 characters
```

### `embeddings.py`

Generates semantic embeddings using:

```text
all-MiniLM-L6-v2
```

Each chunk is represented as a **384-dimensional vector**.

### `retriever.py`

Creates a FAISS vector index and performs similarity search.

The current implementation uses:

```text
FAISS IndexFlatL2
```

### `llm_service.py`

Connects EduGuide to an LLM through OpenRouter.

The LLM receives:

```text
Academic Context
+
Student Question
```

and generates the final response.

### `rag_pipeline.py`

Connects the complete retrieval and generation process:

```text
Question
 ↓
Embedding
 ↓
FAISS Search
 ↓
Top-k Chunks
 ↓
Context
 ↓
LLM
 ↓
Answer
```

### `main.py`

Provides the FastAPI endpoints.

Current endpoints:

```text
GET  /
POST /chat
```

---

## 🧪 Example

### Request

```json
{
  "question": "What is Big Data?"
}
```

### Response

```json
{
  "question": "What is Big Data?",
  "answer": "Based on the academic context, Big Data describes extremely complex, massive datasets..."
}
```

The response is generated using information retrieved from the academic document.

---

## ▶️ Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/aarthi04115/EduGuide.git
cd EduGuide
```

### 2. Create and activate a virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

Install the required packages:

```bash
pip install fastapi uvicorn pymupdf sentence-transformers faiss-cpu openai python-dotenv
```

### 4. Configure the API key

Create a `.env` file in the project root:

```text
OPENROUTER_API_KEY=your_api_key_here
```

**Never commit your `.env` file or expose your API key publicly.**

### 5. Start the backend

From the `backend` directory:

```powershell
cd backend
uvicorn main:app --reload
```

### 6. Open the API documentation

Open:

```text
http://127.0.0.1:8000/docs
```

Use:

```text
POST /chat
```

with:

```json
{
  "question": "What is Big Data?"
}
```

---

## 🧪 Current Testing

The RAG pipeline has been tested using an academic Big Data Analytics PDF.

For the query:

```text
What is Volume in Big Data?
```

FAISS successfully retrieved relevant academic chunks containing information about:

* Volume
* Velocity
* Variety
* Big Data characteristics
* Examples of large-scale data

The retrieved context was then passed to the LLM, which generated an academic answer grounded in the retrieved material.

---

## 🎯 Project Goal

The long-term goal of EduGuide is to evolve from a basic academic RAG chatbot into an **adaptive learning assistant**.

Planned capabilities include:

```text
Student Question
       ↓
Intent Detection
       ↓
Learning Mode
       ↓
Academic Retrieval
       ↓
Context-Aware LLM
       ↓
Personalized Explanation
       ↓
Student Feedback
       ↓
Adaptive Explanation
```

Planned features include:

* Intent detection
* Beginner learning mode
* Intermediate learning mode
* Exam-answer mode
* Interview/viva preparation mode
* Adaptive explanation based on student feedback
* Quiz and practice generation
* Learning-gap detection
* Academic source metadata

These features will be added incrementally as the project develops.

---

## 📌 Current Project Status

### Completed

* [x] Academic PDF text extraction
* [x] Document chunking
* [x] Semantic embeddings
* [x] FAISS vector indexing
* [x] Semantic retrieval
* [x] LLM integration
* [x] RAG pipeline
* [x] FastAPI `/chat` endpoint
* [x] Swagger API testing

### In Development

* [ ] Intent detection
* [ ] Learning modes
* [ ] Adaptive explanation
* [ ] Quiz generation
* [ ] Learning-gap detection
* [ ] Source metadata
* [ ] Persistent vector index
* [ ] Evaluation and performance metrics

---

## 👩‍💻 Project

**EduGuide – Adaptive Academic Learning Assistant**

Built using Python, FastAPI, FAISS, Sentence Transformers, RAG, and LLM technology.

---

## 📄 License

This project is intended for academic and educational purposes.
