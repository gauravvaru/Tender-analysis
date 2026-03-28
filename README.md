# Tender-analysis# Tender Analysis System

A scalable AI-powered system for analyzing tender documents using Retrieval-Augmented Generation (RAG), vector search, and Large Language Models (LLMs). The system extracts structured information such as metadata, eligibility criteria, financial details, and scope of work from PDF tenders.

---

## Overview

This project processes tender documents end-to-end:

1. Ingest PDF documents
2. Split documents into searchable chunks
3. Generate embeddings and store them in a FAISS vector database
4. Retrieve relevant context using semantic search
5. Use an LLM to extract structured information
6. Evaluate system performance using automated tests

---

## Key Features

- Automated PDF ingestion and parsing
- Intelligent text chunking and deduplication
- Vector search using FAISS
- Metadata and scope-of-work extraction using LLMs
- Retrieval evaluation (Precision / Recall / F1)
- Metadata accuracy validation
- Modular and production-ready architecture
- Secure environment variable handling

---

## System Architecture

Documents → Chunking → Embeddings → FAISS Index → Retrieval → LLM → Structured Output

---

## Project Structure

```
Tenders/
│
├── data/
│   └── tenders/                # Input PDF documents
│
├── modules/
│   ├── ingestion.py            # PDF processing
│   ├── chunking.py             # Text segmentation
│   ├── embedding.py            # FAISS vector store
│   ├── retrieval.py            # Search logic
│   └── extraction.py           # LLM interaction
│
├── evaluations/
│   ├── evaluate_metadata.py
│   ├── evaluate_retrieval.py
│   └── compare_chatgpt.py
│
├── tests/
│   └── test_pipeline.py
│
├── app.py                      # Streamlit application
├── ingest.py                   # Document indexing script
├── config.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/gauravvaru/Tender-analysis.git
cd Tender-analysis
```

Create virtual environment:

```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Environment Setup

Create a `.env` file in the project root:

```
LLM_PROVIDER=gemini
LLM_MODEL=models/gemini-2.5-flash
GOOGLE_API_KEY=your_api_key_here
```

Important:

- Never commit `.env` to Git
- Keep API keys secure
- `.gitignore` already protects sensitive files

---

## Running the System

### Step 1 — Add Tender Documents

Place PDF files into:

```
data/tenders/
```

---

### Step 2 — Build the Vector Index

```
python ingest.py
```

---

### Step 3 — Launch the Application

```
streamlit run app.py
```

---

## Running Tests

```
python -m pytest tests/test_pipeline.py -v
```

Expected output:

```
10 passed
```

---

## Evaluation

### Evaluate Metadata Extraction

```
python evaluations/evaluate_metadata.py
```

---

### Evaluate Retrieval Performance

```
python evaluations/evaluate_retrieval.py
```

Metrics:

- Precision
- Recall
- F1 Score

---

### Benchmark Against Another LLM

```
python evaluations/compare_chatgpt.py
```

---

## Security

Sensitive files automatically ignored:

- `.env`
- `.venv`
- API keys
- Model files
- Logs

---

## Typical Workflow

```
Add PDFs → Run ingest.py → Run app.py → Evaluate system
```

---

## Performance Notes

System performance improves with:

- More documents in the dataset
- Better chunking configuration
- Larger vector index
- Hardware acceleration

---

## Requirements

- Python 3.10+
- Git
- Internet connection for model APIs

---

## Maintainer

GitHub:

https://github.com/gauravvaru