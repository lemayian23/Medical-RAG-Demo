# Medical RAG Demo

A domain-specific Retrieval-Augmented Generation system for medical Q&A.

## Tech Stack
- **Embeddings:** MedCPT (NIH-trained on 255M PubMed pairs)
- **Vector Store:** FAISS
- **LLM:** Mistral 7B via Ollama
- **API:** FastAPI
- **Agents:** Router → Retriever → Synthesizer → Critic

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment variables
copy .env.example .env

# 4. Pull Ollama model
ollama pull mistral:7b-instruct-v0.2-q4_0

# 5. Ingest medical documents
python scripts/ingest_documents.py

# 6. Run the server
uvicorn app.main:app --reload

Demo Queries
"What are the treatments for HER2-positive breast cancer?"

"What is the mechanism of action of Tamoxifen?"


---

## Step 1.6: Set Up Virtual Environment & Install Dependencies

Run these commands:

```bash
# Navigate to project
cd D:\Medicine\medical-rag-projects\medical-rag-demo

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy .env
copy .env.example .env

# Pull the Ollama model
ollama pull mistral:7b-instruct-v0.2-q4_0