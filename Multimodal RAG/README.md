# 📚 Fullstack Multimodal RAG — Windows Setup Guide

A full-stack RAG (Retrieval-Augmented Generation) Q&A system that processes PDF documents (text, images, tables) and answers questions using Gemini or Ollama.

---

## 🗂️ Project Structure

```
fullstack-multimodal-rag/
├── START_HERE.bat          ← Run this first (sets up everything + launches app)
├── INGEST_DATA.bat         ← Run this to index the PDF into OpenSearch
├── STOP_SERVICES.bat       ← Run this to stop Docker containers when done
│
├── app.py                  ← Gradio web UI
├── ingestion.py            ← PDF parsing + indexing pipeline
├── chunking.py             ← Text/image/table extraction
├── retrieval.py            ← Keyword, semantic, hybrid search
├── generation.py           ← Gemini / Ollama response generation
├── helper.py               ← Embeddings + OpenSearch client
├── docker-compose.yml      ← OpenSearch + Dashboards setup
├── requirements.txt        ← Python dependencies
├── .env                    ← Your API keys (auto-filled on first run)
├── files/                  ← Put your PDF here
│   └── 2312.10997v5.pdf    ← RAG survey paper (you must download this)
└── dev.ipynb               ← Development/exploration notebook
```

---

## ✅ Prerequisites

Install these **before** running anything:

| Tool | Download | Notes |
|------|----------|-------|
| **Python 3.10+** | https://www.python.org/downloads/ | ✅ Check "Add to PATH" |
| **Docker Desktop** | https://www.docker.com/products/docker-desktop/ | Must be running |
| **Ollama** *(optional)* | https://ollama.com | Only needed for local model |

---

## 🚀 How to Run (Step by Step)

### Step 1 — Get a Gemini API Key (Free)
1. Go to https://aistudio.google.com/apikey
2. Click **"Create API Key"**
3. Copy the key — you'll paste it when prompted

### Step 2 — Download the PDF
Download the RAG survey paper and save it to the `files/` folder:
- URL: https://arxiv.org/pdf/2312.10997
- Save as: `files\2312.10997v5.pdf`

### Step 3 — Run Setup & Launch
Double-click **`START_HERE.bat`**

This will:
- Prompt you for your Gemini API key (once, saves to `.env`)
- Start OpenSearch via Docker
- Create a Python virtual environment
- Install all dependencies
- Launch the Gradio web app at **http://localhost:7860**

### Step 4 — Ingest the PDF (First Time Only)
Before asking questions, you need to index the PDF.  
Double-click **`INGEST_DATA.bat`**

This will:
- Parse the PDF into text chunks, images, and tables
- Generate descriptions for images/tables using Gemini
- Create vector embeddings using Ollama (`nomic-embed-text`)
- Index everything into OpenSearch

⚠️ **This takes 5–20 minutes** on first run.  
After that, you don't need to run it again unless you add new documents.

### Step 5 — Ask Questions!
Open http://localhost:7860 and start asking questions about RAG.

---

## 🔧 Optional: Local Ollama Setup

To use the **Ollama** model option in the UI (runs 100% locally, no internet needed):

```
# In a command prompt, after installing Ollama:
ollama pull nomic-embed-text
ollama pull deepseek-r1:1.5b
ollama serve
```

---

## 🔍 Search Methods Explained

| Method | How it works |
|--------|-------------|
| **Keyword** | Traditional text search — fast, exact matches |
| **Semantic** | Vector similarity — finds conceptually related content |
| **Hybrid** | Combines both — best results (recommended) |

---

## 🛠️ Troubleshooting

**"Docker not found"** → Install Docker Desktop and make sure it's started (whale icon in system tray)

**"Cannot connect to OpenSearch"** → Docker Desktop must be running. Try: `docker compose up -d`

**"GEMINI_API_KEY not set"** → Edit `.env` and replace `YOUR_GEMINI_API_KEY_HERE` with your key

**Gradio won't open** → Make sure port 7860 isn't in use. Try: `netstat -ano | findstr :7860`

**Ingestion is very slow** → Normal for the first run. Image/table descriptions via Gemini take time.

**Ollama embedding error** → Make sure Ollama is running: open a terminal and run `ollama serve`

---

## 📊 Architecture

```
PDF Document
     │
     ▼
[unstructured] ──► Text Chunks ──►
                ──► Images + Captions ──► [Gemini] descriptions ──►  OpenSearch
                ──► Tables ──► [Gemini] descriptions ──►              (vector + text index)
                                                                            │
User Query ──────────────────────────────────────────────────────────────── │
     │                                                                       │
     ▼                                                                       ▼
[Ollama embeddings]                                               [Hybrid/Semantic/Keyword Search]
                                                                            │
                                                                            ▼
                                                                   [Gemini / Ollama LLM]
                                                                            │
                                                                            ▼
                                                                      Answer ◄── Gradio UI
```

---

## 📝 Notes

- The `.env` file stores your API key — **never share this file**
- OpenSearch data persists in a Docker volume across restarts
- The `files/` folder is where you put additional PDFs (modify `ingestion.py` to index them)
- OpenSearch Dashboards are available at http://localhost:5601 for exploring indexed data
