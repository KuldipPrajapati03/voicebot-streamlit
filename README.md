# 📚 Enterprise RAG Assistant

A Streamlit-based **PDF RAG (Retrieval-Augmented Generation) chatbot** that lets you upload PDF documents, ask questions about their content, and get cited answers — powered by **Groq LLMs**, **FAISS** vector search, and **HuggingFace embeddings**. It also supports **image-based Q&A** using a vision-capable Groq model.

### 🔗 Live Demo
Try it here: **[chatbot-kuldip-prajapati.streamlit.app](https://chatbot-kuldip-prajapati.streamlit.app/)**

---

## ✨ Features

- **📂 PDF Knowledge Base** — Upload one or more PDFs and build a searchable knowledge base.
- **🧠 Smart Chunking** — Uses `unstructured` for layout-aware PDF parsing (tables, titles, text) combined with recursive text splitting for optimal chunk sizes.
- **🔍 Semantic Search** — FAISS vector store with `all-MiniLM-L6-v2` sentence embeddings for fast, relevant retrieval.
- **🤖 Conversational RAG Pipeline** — Built with **LangGraph** (`retrieve` → `generate`) and Groq's `openai/gpt-oss-120b` model, with automatic follow-up question resolution using recent chat history.
- **📌 Source Citations** — Every answer cites the source document and page number(s), or falls back to general AI knowledge when no relevant context is found.
- **🖼️ Image Q&A** — Attach an image directly in the chat input and ask questions about it (text, charts, diagrams, screenshots, etc.) using a vision-capable Groq model.
- **📄 PDF Chunk Viewer** — Inspect exactly how each PDF was chunked, with a one-click "Copy Text" option, via an in-app modal.
- **🗂️ Document Management** — View all processed/active documents, and delete individual PDFs (or clear the entire vector store) from a persistent configuration drawer.
- **💬 Chat-style UI** — Custom-styled chat bubbles, action buttons (copy/like) on responses, and a clean, modern light theme.

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────┐      ┌──────────────────┐
│  Retrieve    │ ───▶ │  FAISS Vector    │
│  Node        │      │  Store (top-k=4) │
└─────────────┘      └──────────────────┘
     │
     ▼
┌─────────────┐      ┌──────────────────┐
│  Generate    │ ───▶ │  Groq LLM        │
│  Node        │      │  (gpt-oss-120b)  │
└─────────────┘      └──────────────────┘
     │
     ▼
Answer + Source Citations
```

- **Framework:** [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` with two nodes: `retrieve` and `generate`.
- **Chat Memory:** The last 3 completed Q&A pairs are passed into every prompt so the assistant can resolve follow-up/contextual questions (e.g., "what about it?", "the same as above").
- **Vector Store:** FAISS, persisted locally to disk (`faiss_index_store_new/`) so the knowledge base survives app restarts.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM Orchestration | LangGraph + LangChain |
| LLM Provider | Groq (`langchain-groq`) |
| Text Model | `openai/gpt-oss-120b` |
| Vision Model | `qwen/qwen3.6-27b` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (via `sentence-transformers`) |
| Vector Store | FAISS |
| PDF Parsing | `unstructured` (with `unstructured_inference`, OCR via Tesseract) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |

---

## 📋 Prerequisites

### System dependencies
These are required by `unstructured` for PDF parsing, OCR, and layout detection:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils libmagic-dev libreoffice libgl1
```

### Python dependencies
All Python packages are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

### API Key
You'll need a **Groq API key**. Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Install system dependencies** (see [Prerequisites](#-prerequisites))

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your `.env` file** with your `GROQ_API_KEY`

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

6. Open the app in your browser (usually `http://localhost:8501`)

---

## 💡 Usage

1. Click the **Config** button (top-right corner) to open the configuration drawer.
2. Upload one or more PDF files under **Knowledge Base** and click **Process**.
3. Once indexed, start asking questions in the chat input at the bottom.
4. Click on any processed document under **Active Context** to inspect its parsed chunks.
5. To ask a question about an image instead, attach an image directly in the chat input box.
6. Use **Clear Store** to wipe the entire knowledge base, or the **Manage Documents** dropdown to remove a single PDF.

---

## 📁 Project Structure

```
.
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── packages.txt / apt deps    # System-level dependencies (Tesseract, Poppler, etc.)
├── .env                       # Environment variables (GROQ_API_KEY) — not committed
└── faiss_index_store_new/     # Local FAISS vector store (created at runtime)
```

---

## ⚠️ Notes & Limitations

- The FAISS index is stored **locally on disk**, so on platforms with ephemeral storage (like Streamlit Community Cloud), the knowledge base may reset when the app restarts or redeploys.
- OCR/layout inference depends on `unstructured_inference`, Tesseract, and Poppler being correctly installed on the host system.
- Answers are only as good as the retrieved context — for very large or heavily image-based PDFs, consider adjusting the chunking parameters (`max_characters`, `new_after_n_chars`, `combine_text_under_n_chars`) in `get_pdf_documents()`.

---

## 👤 Developer

**Kuldip Prajapati**
[LinkedIn](https://www.linkedin.com/in/kuldip-prajapati)

