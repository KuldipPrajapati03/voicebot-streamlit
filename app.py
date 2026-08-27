import os
import uuid
import ast
import shutil
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")

try:
    st._config.set_option("theme.base", "light")
    st._config.set_option("theme.primaryColor", "#4f46e5")
    st._config.set_option("theme.backgroundColor", "#f8fafc")
    st._config.set_option("theme.secondaryBackgroundColor", "#f1f5f9")
    st._config.set_option("theme.textColor", "#0f172a")
except Exception:
    pass

# --- Unstructured & Chunking Imports ---
from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- VectorStore & Embeddings ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- LLM & LangGraph Imports ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import List
from typing_extensions import TypedDict


# ==========================================
# 1. Text Extraction & Vectorstore
# ==========================================

def get_pdf_documents(file):
    """Extract elements from uploaded PDF file and return Documents with metadata (source & page number)."""
    temp_file = f"temp_{uuid.uuid4()}.pdf"
    with open(temp_file, "wb") as f:
        f.write(file.getvalue())

    with st.spinner("Extracting text from PDF..."):
        elements = partition_pdf(
            filename=temp_file,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
        )
    os.remove(temp_file)

    with st.spinner("Chunking document..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_chunks = []

        for element in elements:
            page_num = element.metadata.page_number if hasattr(element,
                                                               "metadata") and element.metadata.page_number else 1
            element_id = getattr(element, "id", str(uuid.uuid4()))

            splits = text_splitter.split_text(element.text)
            for chunk in splits:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": file.name,
                        "page": page_num,
                        "element_id": str(element_id)
                    }
                )
                all_chunks.append(doc)

    return all_chunks


def get_vectorstore(_text_chunks, store_name, _embeddings):
    with st.spinner("Saving chunks to vectorstore & generating embeddings..."):
        if st.session_state.vectorstore is not None:
            vectorstore = st.session_state.vectorstore
            vectorstore.add_documents(_text_chunks)
        elif os.path.exists(store_name):
            vectorstore = FAISS.load_local(store_name, _embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_documents(_text_chunks)
        else:
            vectorstore = FAISS.from_documents(_text_chunks, _embeddings)
        vectorstore.save_local(store_name)
    return vectorstore


# ==========================================
# 2. LangGraph State & RAG Node Definitions
# ==========================================

class State(TypedDict):
    question: str
    context_with_ids: str
    retrieved_docs: List[Document]
    response: str


def retrieve_node(state: State):
    """Node that retrieves relevant documents from the vectorstore."""
    if st.session_state.get("vectorstore") is None:
        return {"context_with_ids": "", "retrieved_docs": []}

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(state["question"])

    context_with_ids = "\n\n".join(
        [f"ID: {doc.metadata.get('element_id', 'N/A')}\nContent: {doc.page_content}" for doc in docs]
    )
    return {"context_with_ids": context_with_ids, "retrieved_docs": docs}


def generate_node(state: State):
    """Node that evaluates context, falls back to general knowledge if needed, and formats sources."""
    prompt_template = ChatPromptTemplate.from_template(
        """You are an AI assistant designed to answer questions based on the provided document context.

Analyze the following context, which consists of text chunks associated with unique identifiers (IDs):

Context:
{context_with_ids}

Answer the following Question:
Question: {question}

Instructions for Formulating Your Response:

1. **Primary Strategy (Context-Based):**
   - Read the provided Context carefully.
   - If the context contains direct, high-quality, and relevant information to answer the user's question, base your answer strictly and exclusively on that context.
   - Use precise wording from the context without adding rephrasing or summaries. Do not assume or extrapolate.

2. **Fallback Strategy (General Knowledge):**
   - If the Context does NOT contain relevant information to answer the question, or if there is no relevant context material, ignore the context completely.
   - In this case, provide a helpful answer using your general knowledge as an AI model.

3. **Source Citation Rule (Crucial):**
   - After your answer, on a new line, write "Source:" followed by:
     - If you used information from the provided context chunks: list the relevant element IDs in list format (e.g., "Source: ['123456']" or "Source: ['123456', '789101']").
     - If the context had no relevant information and you relied on general AI knowledge: write "Source: None".

Answer:"""
    )

    chain = prompt_template | st.session_state.llm
    res = chain.invoke({"context_with_ids": state["context_with_ids"], "question": state["question"]})
    result_text = res.content

    if "\nSource:" in result_text:
        answer_parts = result_text.rsplit("\nSource:", 1)
        answer = answer_parts[0].strip()
        source_type = answer_parts[1].strip()
    else:
        answer = result_text.strip()
        source_type = "None"

    sources_pages = {}
    if source_type.startswith("['") and source_type.endswith("']"):
        try:
            source_element_ids = ast.literal_eval(source_type)
            for doc in state["retrieved_docs"]:
                if doc.metadata.get("element_id") in source_element_ids:
                    src = doc.metadata.get("source", "Unknown Document")
                    page = doc.metadata.get("page", "N/A")
                    if src not in sources_pages:
                        sources_pages[src] = set()
                    sources_pages[src].add(page)
        except (ValueError, SyntaxError):
            sources_pages = {}

    if sources_pages:
        citation_strings = []
        for src, pages in sources_pages.items():
            sorted_pages = sorted(list(pages))
            pages_str = ", ".join(map(str, sorted_pages))
            citation_strings.append(f"**{src}** (Page {pages_str})")

        formatted_sources = " | ".join(citation_strings)
        final_response = f"{answer}\n\n**Sources:** {formatted_sources}"
    else:
        final_response = f"{answer}\n\n*(AI knowledge)*"

    return {"response": final_response}


def build_rag_graph():
    """Builds and compiles the LangGraph state graph."""
    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", retrieve_node)
    graph_builder.add_node("generate", generate_node)

    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", END)

    return graph_builder.compile()


# Dialog to display document text chunks
@st.dialog("PDF Chunks Viewer", width="large")
def show_pdf_chunks(pdf_name):
    vectorstore = st.session_state.vectorstore
    if vectorstore is None:
        st.write("Vectorstore is empty.")
        return

    doc_dict = vectorstore.docstore._dict
    chunks_info = []

    for doc_id, doc in doc_dict.items():
        src = doc.metadata.get("source")
        if src == pdf_name:
            page = doc.metadata.get("page", 1)
            chunks_info.append((page, doc.page_content))

    chunks_info.sort(key=lambda x: x[0])

    col1, col2, col3 = st.columns([5, 2, 2])
    with col1:
        st.markdown(
            f"<h3 style='margin: 0px; padding: 0px; font-size: 18px; color: var(--text-primary); font-weight: 600;'>"
            f"Chunk Details <span style='font-size: 13px; font-weight: normal; color: var(--text-secondary); margin-left: 8px;'>"
            f"({len(chunks_info)} chunks for <code style='color: var(--accent-primary); background: var(--bg-tertiary); padding:2px 6px; border-radius:4px;'>{pdf_name}</code>)</span></h3>",
            unsafe_allow_html=True
        )
    with col2:
        expand_all = st.checkbox("Expand All", value=False, key="expand_all_chunks_checkbox")
    with col3:
        all_text = "\n\n".join([text for _, text in chunks_info])
        import json
        escaped_text = json.dumps(all_text)
        copy_button_html = f"""
        <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%;">
            <button id="copyBtn" class="copy-btn" onclick="copyToClip()">📋 Copy Text</button>
        </div>
        <script>
        function isParentDark() {{
            try {{
                const doc = window.parent.document;
                const theme = doc.documentElement.getAttribute('data-theme');
                if (theme) return theme === 'dark';
                return window.parent.matchMedia && window.parent.matchMedia('(prefers-color-scheme: dark)').matches;
            }} catch (e) {{
                return false;
            }}
        }}

        (function applyTheme() {{
            const dark = isParentDark();
            const btn = document.getElementById("copyBtn");
            if (dark) {{
                btn.style.backgroundColor = "#1e293b";
                btn.style.color = "#f1f5f9";
                btn.style.borderColor = "#334155";
            }}
        }})();

        function copyToClip() {{
            const text = {escaped_text};
            function handleSuccess() {{
                const btn = document.getElementById("copyBtn");
                btn.innerHTML = "✓ Copied!";
                btn.style.backgroundColor = "#22c55e";
                btn.style.color = "#ffffff";
                btn.style.borderColor = "#22c55e";
                setTimeout(function() {{
                    btn.innerHTML = "📋 Copy Text";
                    applyTheme();
                }}, 2000);
            }}
            if (window.parent && window.parent.navigator && window.parent.navigator.clipboard) {{
                window.parent.navigator.clipboard.writeText(text).then(handleSuccess).catch(function(err) {{
                    const textarea = document.createElement("textarea");
                    textarea.value = text;
                    document.body.appendChild(textarea);
                    textarea.select();
                    document.execCommand("copy");
                    document.body.removeChild(textarea);
                    handleSuccess();
                }});
            }} else {{
                const textarea = document.createElement("textarea");
                textarea.value = text;
                document.body.appendChild(textarea);
                textarea.select();
                document.execCommand("copy");
                document.body.removeChild(textarea);
                handleSuccess();
            }}
        }}
        </script>
        <style>
        .copy-btn {{
            border-radius: 8px !important;
            font-weight: 500 !important;
            font-size: 0.875rem !important;
            padding: 10px 10px;
            border: 1px solid #e2e8f0 !important;
            background-color: #f8fafc !important;
            color: #0f172a !important;
            cursor: pointer;
            transition: all 0.2s ease-in-out;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            width: 100%;
            text-align: center;
            box-sizing: border-box;
        }}
        .copy-btn:hover {{
            background-color: #f1f5f9 !important;
            border-color: #4f46e5 !important;
        }}
        </style>
        """
        import streamlit.components.v1 as components
        components.html(copy_button_html, height=45)

    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    with st.container(height=450):
        for idx, (page, text) in enumerate(chunks_info):
            with st.expander(f"📄 Chunk {idx + 1} (Page {page})", expanded=expand_all):
                st.text_area(
                    label=f"Chunk {idx + 1} Content",
                    value=text,
                    height=150,
                    disabled=True,
                    key=f"dialog_chunk_{idx}",
                    label_visibility="collapsed"
                )


# ==========================================
# 3. Streamlit Interface Setup
# ==========================================

st.set_page_config(
    page_title="Enterprise PDF RAG Workspace",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

chat_css = """
<style>
/* --- Global Typography & Theme Resets & Color Tokens --- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-primary: #f8fafc;
    --bg-secondary: #ffffff;
    --bg-tertiary: #f1f5f9;
    --bg-sidebar: #ffffff;
    --bg-card: #ffffff;
    --bg-input: #f8fafc;
    --bg-hover: #f1f5f9;
    --bg-chat-user: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
    --bg-chat-assistant: #ffffff;
    --border-primary: #e2e8f0;
    --border-secondary: #cbd5e1;
    --border-focus: #6366f1;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #94a3b8;
    --text-inverse: #ffffff;
    --accent-primary: #4f46e5;
    --accent-secondary: #6366f1;
    --accent-hover: #4338ca;
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
    --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
    --radius-sm: 6px;
    --radius-md: 8px;
    --radius-lg: 12px;
    --radius-xl: 16px;
}

/* ============================================
   DARK THEME OVERRIDES (Option 2)
   Only reachable when FORCE_LIGHT_THEME = False
   ============================================ */
[data-theme="dark"] {
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-tertiary: #1e293b;
    --bg-sidebar: #0f172a;
    --bg-card: #1e293b;
    --bg-input: #1e293b;
    --bg-hover: #334155;
    --bg-chat-user: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
    --bg-chat-assistant: #1e293b;
    --border-primary: #334155;
    --border-secondary: #475569;
    --border-focus: #818cf8;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --text-inverse: #ffffff;
    --accent-primary: #818cf8;
    --accent-secondary: #6366f1;
    --accent-hover: #a5b4fc;
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.3);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.4), 0 2px 4px -2px rgb(0 0 0 / 0.4);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.4), 0 4px 6px -4px rgb(0 0 0 / 0.4);
    --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.5), 0 8px 10px -6px rgb(0 0 0 / 0.5);
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* --- Developer Info Styling --- */
.developer-info {
    margin-bottom: 8px;
}

.developer-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    font-weight: 600;
}

.developer-name {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin-top: 2px;
}

.developer-contact {
    font-size: 12px;
    color: var(--accent-primary);
    text-decoration: none;
}

.developer-contact:hover {
    text-decoration: underline;
}

/* --- Professional Header Banner Styling --- */
header[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.85) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-bottom: 1px solid var(--border-primary) !important;
    height: 3.75rem !important;
}

[data-theme="dark"] header[data-testid="stHeader"] {
    background: rgba(15, 23, 42, 0.85) !important;
}

header[data-testid="stHeader"]::before {
    content: "⚡ Enterprise RAG Assistant";
    position: absolute;
    left: 1.5rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--text-primary) !important;
}

/* --- Refined Sidebar Styling --- */
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-primary) !important;
}

[data-testid="stSidebar"] hr {
    border-color: var(--border-primary) !important;
    margin: 1.25rem 0 !important;
}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    font-weight: 600 !important;
}

/* --- Input Fields & Form Controls --- */
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] > div > div {
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border-primary) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-md) !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stTextInput"] input:focus {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

div[data-testid="stWidgetLabel"] p {
    color: var(--text-secondary) !important;
    font-size: 0.825rem !important;
    font-weight: 500 !important;
}

/* --- File Uploader Styling --- */
div[data-testid="stFileUploader"] {
    background-color: var(--bg-input) !important;
    border: 2px dashed var(--border-secondary) !important;
    border-radius: var(--radius-lg) !important;
    padding: 10px !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-primary) !important;
    background-color: rgba(99, 102, 241, 0.05) !important;
}

/* --- Button Modernization --- */
div.stButton > button {
    border-radius: var(--radius-md) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    transition: all 0.2s ease-in-out !important;
    border: 1px solid var(--border-primary) !important;
    background-color: var(--bg-input) !important;
    color: var(--text-primary) !important;
}

div.stButton > button:hover {
    background-color: var(--bg-hover) !important;
    border-color: var(--accent-primary) !important;
    color: var(--text-primary) !important;
}

/* Primary Action Buttons */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    border: none !important;
    color: var(--text-inverse) !important;
    box-shadow: var(--shadow-sm) !important;
}

div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%) !important;
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px);
}

/* --- Chat Container & Messages Styling --- */
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] span {
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}

/* Transparent wrapper structure */
div[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 10px 0px !important;
}

div[data-testid="stChatMessage"] > div:nth-child(2) {
    width: fit-content !important;
    max-width: 82% !important;
    flex-grow: 0 !important;
}

/* Right side user alignment */
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse !important;
}

div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div:nth-child(2) {
    margin-left: auto !important;
    margin-right: 12px !important;
}

/* User Message Bubble */
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] {
    background: var(--bg-chat-user) !important;
    color: var(--text-inverse) !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 12px 18px !important;
    box-shadow: var(--shadow-sm) !important;
}

/* Assistant Message Bubble */
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) > div:nth-child(2) {
    margin-right: auto !important;
    margin-left: 12px !important;
}

div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stChatMessageContent"] {
    background-color: var(--bg-chat-assistant) !important;
    border: 1px solid var(--border-primary) !important;
    color: var(--text-primary) !important;
    border-radius: 16px 16px 16px 4px !important;
    padding: 14px 20px !important;
    box-shadow: var(--shadow-sm) !important;
}

/* --- Chat Input Control Modernization --- */
div[data-testid="stChatInput"] {
    background-color: var(--bg-secondary) !important;
    border: 1px solid var(--border-primary) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow-md) !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stChatInput"]:focus-within {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1), var(--shadow-md) !important;
}

div[data-testid="stChatInput"] textarea {
    color: var(--text-primary) !important;
    font-size: 0.95rem !important;
}

/* --- Dialog Modals Clean Up --- */
div[role="dialog"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-primary) !important;
    border-radius: var(--radius-xl) !important;
    box-shadow: var(--shadow-xl) !important;
}

div[data-testid*="DialogHeader"] {
    display: none !important;
}

div[role="dialog"] button[kind="header"] svg {
    color: var(--text-primary) !important;
}

/* --- Welcome State (Empty Chat) --- */
.welcome-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 50vh;
    text-align: center;
    color: var(--text-muted);
    font-family: 'Inter', sans-serif;
    margin-top: 5vh;
}

.welcome-icon {
    font-size: 48px;
    margin-bottom: 16px;
    opacity: 0.8;
}

.welcome-title {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 8px;
}

.welcome-subtitle {
    font-size: 14px;
    max-width: 400px;
    line-height: 1.6;
    color: var(--text-secondary);
}

/* --- Elements not covered above, needed for a clean dark theme too --- */

/* st.warning / st.success / st.error boxes */
[data-testid="stAlert"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-primary) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stAlert"] p {
    color: var(--text-primary) !important;
}

/* st.expander (chunk viewer) */
div[data-testid="stExpander"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-primary) !important;
    border-radius: var(--radius-md) !important;
}
div[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
}
div[data-testid="stExpander"] summary:hover {
    color: var(--accent-primary) !important;
}

/* st.text_area (chunk content display) */
textarea {
    background-color: var(--bg-input) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-primary) !important;
}
textarea:disabled {
    -webkit-text-fill-color: var(--text-primary) !important;
    opacity: 1 !important;
}

/* st.checkbox label ("Expand All") */
label[data-baseweb="checkbox"] span {
    color: var(--text-secondary) !important;
}

/* st.selectbox closed box text */
div[data-testid="stSelectbox"] span {
    color: var(--text-primary) !important;
}

/* st.selectbox dropdown popover — renders in a portal outside the input */
div[data-baseweb="popover"] ul,
div[data-baseweb="menu"],
div[data-baseweb="popover"] div {
    background-color: var(--bg-card) !important;
}
div[data-baseweb="popover"] li,
div[data-baseweb="menu"] li {
    color: var(--text-primary) !important;
}
div[data-baseweb="popover"] li:hover,
div[data-baseweb="menu"] li:hover {
    background-color: var(--bg-hover) !important;
}

/* Main app text fallback (headers, captions, plain markdown) */
.stApp p, .stApp span, .stApp label {
    color: var(--text-primary);
}
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text-muted) !important;
}

/* Spinner text */
.stSpinner > div {
    color: var(--text-secondary) !important;
}
</style>
"""

chat_css = chat_css.replace(
    "</style>",
    """
#MainMenu {
    visibility: hidden !important;
}
[data-testid="stToolbar"] {
    visibility: hidden !important;
}
</style>"""
)

st.html(chat_css)

# Reopen sidebar script via header settings button (⚙️)
reopen_sidebar_js = """
<script>
(function() {
    let parentDoc = document;
    try {
        if (window.parent && window.parent.document && window.parent.document.body) {
            parentDoc = window.parent.document;
        }
    } catch (e) {
        parentDoc = document;
    }

    function updateSidebarToggleUI() {
        try {
            const header = parentDoc.querySelector('header[data-testid="stHeader"]');
            if (!header) return;

            let button = parentDoc.getElementById("customSettingsBtn");
            if (!button) {
                button = parentDoc.createElement("button");
                button.id = "customSettingsBtn";
                button.innerHTML = "⚙️";
                button.style.position = "absolute";
                button.style.left = "1.5rem";
                button.style.top = "50%";
                button.style.transform = "translateY(-50%)";
                button.style.background = "none";
                button.style.border = "none";
                button.style.cursor = "pointer";
                button.style.fontSize = "1.25rem";
                button.style.zIndex = "999999";
                button.style.padding = "4px";
                button.style.display = "none";
                button.style.alignItems = "center";
                button.style.justifyContent = "center";
                button.style.color = "var(--text-primary)";
                button.style.transition = "opacity 0.2s ease, transform 0.2s ease";

                button.onmouseover = function() {
                    button.style.opacity = "0.7";
                    button.style.transform = "translateY(-50%) scale(1.1)";
                };
                button.onmouseout = function() {
                    button.style.opacity = "1.0";
                    button.style.transform = "translateY(-50%) scale(1.0)";
                };

                button.onclick = function() {
                    const collapsedBtn = parentDoc.querySelector('[data-testid="stSidebarCollapsedControl"] button') ||
                                         parentDoc.querySelector('[data-testid="collapsedSidebar"] button') || 
                                         parentDoc.querySelector('[data-testid="stSidebarCollapsedControl"]') ||
                                         parentDoc.querySelector('[data-testid="collapsedSidebar"]') ||
                                         parentDoc.querySelector('button[aria-label="Open sidebar"]');
                    if (collapsedBtn) {
                        collapsedBtn.click();
                    } else {
                        const closeBtn = parentDoc.querySelector('[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button') ||
                                         parentDoc.querySelector('[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"]') ||
                                         parentDoc.querySelector('button[aria-label="Close sidebar"]');
                        if (closeBtn) {
                            closeBtn.click();
                        }
                    }
                };

                header.appendChild(button);
            } else if (button.parentNode !== header) {
                header.appendChild(button);
            }

            // Robust check if sidebar is collapsed (closed)
            const sidebar = parentDoc.querySelector('[data-testid="stSidebar"]');
            const collapsedControl = parentDoc.querySelector('[data-testid="stSidebarCollapsedControl"]') || 
                                     parentDoc.querySelector('[data-testid="collapsedSidebar"]') ||
                                     parentDoc.querySelector('button[aria-label="Open sidebar"]');

            let isClosed = false;
            if (collapsedControl) {
                isClosed = true;
            } else if (sidebar) {
                const ariaExpanded = sidebar.getAttribute('aria-expanded');
                if (ariaExpanded === 'false') {
                    isClosed = true;
                } else {
                    const rect = sidebar.getBoundingClientRect();
                    if (rect.width === 0 || rect.left < 0) {
                        isClosed = true;
                    }
                }
            } else {
                isClosed = true;
            }

            // Obtain or create custom header offset style
            let styleEl = parentDoc.getElementById("customHeaderStyle");
            if (!styleEl) {
                styleEl = parentDoc.createElement("style");
                styleEl.id = "customHeaderStyle";
                parentDoc.head.appendChild(styleEl);
            } else if (styleEl.parentNode !== parentDoc.head) {
                parentDoc.head.appendChild(styleEl);
            }

            if (isClosed) {
                button.style.display = "flex";
                styleEl.innerHTML = `
                    header[data-testid="stHeader"]::before {
                        left: 3.5rem !important;
                    }
                `;
            } else {
                button.style.display = "none";
                styleEl.innerHTML = `
                    header[data-testid="stHeader"]::before {
                        left: 1.5rem !important;
                    }
                `;
            }
        } catch (e) {
            console.error("Error managing custom settings button:", e);
        }
    }

    updateSidebarToggleUI();
    setInterval(updateSidebarToggleUI, 200);
})();
</script>
"""
# st.html(reopen_sidebar_js, unsafe_allow_javascript=True)
components.html(reopen_sidebar_js, height=0)

# Intercept close/reload window if data active in session
if st.session_state.get("vectorstore") is not None:
    prevent_close_js = """
    <script>
        if (!window._closeHandler) {
            window._closeHandler = function (e) {
                e.preventDefault();
                e.returnValue = 'Please clear vectorstore using the provided button before exit.';
                return e.returnValue;
            };
        }
        window.removeEventListener('beforeunload', window._closeHandler);
        window.addEventListener('beforeunload', window._closeHandler);
    </script>
    """
    # st.html(prevent_close_js, unsafe_allow_javascript=True)
    components.html(prevent_close_js, height=0)

else:
    remove_close_js = """
    <script>
        if (window._closeHandler) {
            window.removeEventListener('beforeunload', window._closeHandler);
        }
    </script>
    """
    # st.html(remove_close_js, unsafe_allow_javascript=True)
    components.html(remove_close_js, height=0)


store_dir = "faiss_index_store_new"

# Extract unique file sources from FAISS vectorstore docstore
processed_files = []
if st.session_state.get("vectorstore") is not None:
    try:
        sources = set()
        for doc in st.session_state.vectorstore.docstore._dict.values():
            src = doc.metadata.get("source")
            if src:
                sources.add(src)
        processed_files = sorted(list(sources))
    except Exception:
        processed_files = []

# Sidebar Layout & Configuration
with st.sidebar:
    st.markdown(
        """
        <div class="developer-info">
            <div class="developer-label">Developed By</div>
            <a href="https://www.linkedin.com/in/kuldip-prajapati" target="_blank" class="developer-name">Kuldip Prajapati</a>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.divider()

    st.header("📂 Knowledge Base")
    uploaded_file = st.file_uploader("Upload Document", type=["pdf"], accept_multiple_files=True)

    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("Process", use_container_width=True, type="primary")
    with col2:
        clear_btn = st.button("Clear Store", use_container_width=True)

    if processed_files:
        st.divider()
        st.header("📄 Active Context")
        st.caption("Click a file to inspect parsed chunks:")
        for i, pdf_name in enumerate(processed_files):
            if st.button(f"📄 {pdf_name}", key=f"view_pdf_{i}", use_container_width=True):
                show_pdf_chunks(pdf_name)

        st.write("")
        pdf_to_delete = st.selectbox(
            "Manage Documents:",
            ["-- Select file to remove --"] + processed_files,
            key="pdf_to_delete_selectbox"
        )
        if pdf_to_delete != "-- Select file to remove --":
            if st.button("Delete Document", use_container_width=True, type="primary"):
                vectorstore = st.session_state.vectorstore
                doc_ids_to_delete = []
                for doc_id, doc in list(vectorstore.docstore._dict.items()):
                    if doc.metadata.get("source") == pdf_to_delete:
                        doc_ids_to_delete.append(doc_id)

                if doc_ids_to_delete:
                    if len(doc_ids_to_delete) >= len(vectorstore.docstore._dict):
                        if os.path.exists(store_dir):
                            shutil.rmtree(store_dir, ignore_errors=True)
                        st.session_state.vectorstore = None
                        st.session_state.messages = []
                    else:
                        vectorstore.delete(doc_ids_to_delete)
                        vectorstore.save_local(store_dir)
                    st.success(f"Removed: {pdf_to_delete}")
                    st.rerun()

# Session State & Model Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "embeddings" not in st.session_state:
    with st.spinner("Initializing embedding engine..."):
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if "vectorstore" not in st.session_state:
    if os.path.exists(store_dir):
        try:
            st.session_state.vectorstore = FAISS.load_local(store_dir, st.session_state.embeddings,
                                                            allow_dangerous_deserialization=True)
        except Exception:
            st.session_state.vectorstore = None
    else:
        st.session_state.vectorstore = None

# Process PDF Upload Handlers
if process_btn and uploaded_file:
    chunks = []
    for file in uploaded_file:
        chunks.extend(get_pdf_documents(file))
    vectorstore = get_vectorstore(chunks, store_dir, st.session_state.embeddings)
    st.session_state.vectorstore = vectorstore
    st.success("Documents indexed successfully!")
    st.rerun()

# Handle Clear Vectorstore Action
if clear_btn:
    if os.path.exists(store_dir):
        shutil.rmtree(store_dir, ignore_errors=True)
    st.session_state.vectorstore = None
    st.session_state.messages = []
    st.success("Vectorstore cleared.")
    st.rerun()

# Handle retry action from query params (Action Buttons)
try:
    q_params = st.query_params
    if "retry" in q_params:
        retry_idx = int(q_params["retry"])
        del st.query_params["retry"]
        if "messages" in st.session_state and retry_idx < len(st.session_state.messages):
            user_msg_idx = retry_idx - 1
            if user_msg_idx >= 0 and user_msg_idx < len(st.session_state.messages):
                question = st.session_state.messages[user_msg_idx]["content"]
                if groq_api_key:
                    st.session_state.llm = ChatGroq(
                        groq_api_key=groq_api_key,
                        model_name="openai/gpt-oss-120b",
                        temperature=0.2
                    )
                    st.session_state.graph = build_rag_graph()
                    initial_state = {"question": question}
                    output = st.session_state.graph.invoke(initial_state)
                    response_text = output.get("response", "No answer could be generated.")
                    st.session_state.messages[retry_idx]["content"] = response_text
                    st.rerun()
except Exception:
    try:
        import streamlit as _st

        ext_params = _st.experimental_get_query_params()
        if "retry" in ext_params:
            retry_idx = int(ext_params["retry"][0])
            _st.experimental_set_query_params()
            if "messages" in _st.session_state and retry_idx < len(_st.session_state.messages):
                user_msg_idx = retry_idx - 1
                if user_msg_idx >= 0 and user_msg_idx < len(_st.session_state.messages):
                    question = _st.session_state.messages[user_msg_idx]["content"]
                    if groq_api_key:
                        _st.session_state.llm = ChatGroq(
                            groq_api_key=groq_api_key,
                            model_name="openai/gpt-oss-120b",
                            temperature=0.2
                        )
                        _st.session_state.graph = build_rag_graph()
                        initial_state = {"question": question}
                        output = _st.session_state.graph.invoke(initial_state)
                        response_text = output.get("response", "No answer could be generated.")
                        _st.session_state.messages[retry_idx]["content"] = response_text
                        _st.rerun()
    except Exception:
        pass


def get_plain_text(content):
    if "\n\n**Sources:**" in content:
        return content.split("\n\n**Sources:**")[0].strip()
    if "\n\n*(Answer generated using general AI knowledge)*" in content:
        return content.split("\n\n*(AI knowledge)*")[0].strip()
    return content.strip()


def render_action_buttons(content, idx):
    plain_text = get_plain_text(content)
    import json
    escaped_plain_text = json.dumps(plain_text)
    action_btn_html = f"""
    <div class="message-actions">
        <button id="copyBtn" class="action-btn" title="Copy text response" onclick="copyText()">
            <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
        </button>
        <button id="likeBtn" class="action-btn" title="Like response" onclick="toggleLike()" data-liked="false">
            <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
            </svg>
        </button>
    </div>
    <script>
    function isParentDark() {{
        try {{
            const doc = window.parent.document;
            const theme = doc.documentElement.getAttribute('data-theme');
            if (theme) return theme === 'dark';
            return window.parent.matchMedia && window.parent.matchMedia('(prefers-color-scheme: dark)').matches;
        }} catch (e) {{
            return false;
        }}
    }}

    const dark = isParentDark();
    const mutedColor = dark ? "#64748b" : "#94a3b8";
    const hoverBg = dark ? "#334155" : "#f1f5f9";
    document.querySelectorAll(".action-btn").forEach(function(b) {{
        b.style.color = mutedColor;
    }});
    const styleTag = document.createElement("style");
    styleTag.innerHTML = ".action-btn:hover {{ background-color: " + hoverBg + " !important; }}";
    document.head.appendChild(styleTag);

    const msgId = "like_state_" + {idx};
    const textToCopy = {escaped_plain_text};

    const liked = localStorage.getItem(msgId) === "true";
    const btn = document.getElementById("likeBtn");
    if (liked) {{
        btn.setAttribute("data-liked", "true");
        btn.style.color = "#4f46e5";
        btn.querySelector("svg").setAttribute("fill", "#4f46e5");
        btn.querySelector("svg").setAttribute("stroke", "#4f46e5");
    }}

    function copyText() {{
        if (window.parent && window.parent.navigator && window.parent.navigator.clipboard) {{
            window.parent.navigator.clipboard.writeText(textToCopy).then(function() {{
                const btn = document.getElementById("copyBtn");
                btn.innerHTML = `<svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
                setTimeout(function() {{
                    btn.innerHTML = `<svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>`;
                }}, 2000);
            }}).catch(function(err) {{
                const textarea = document.createElement("textarea");
                textarea.value = textToCopy;
                document.body.appendChild(textarea);
                textarea.select();
                document.execCommand("copy");
                document.body.removeChild(textarea);

                const btn = document.getElementById("copyBtn");
                btn.innerHTML = `<svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
                setTimeout(function() {{
                    btn.innerHTML = `<svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>`;
                }}, 2000);
            }});
        }}
    }}

    function toggleLike() {{
        const btn = document.getElementById("likeBtn");
        const liked = btn.getAttribute("data-liked") === "true";
        if (liked) {{
            btn.setAttribute("data-liked", "false");
            btn.style.color = mutedColor;
            btn.querySelector("svg").removeAttribute("fill");
            btn.querySelector("svg").setAttribute("stroke", "currentColor");
            localStorage.setItem(msgId, "false");
        }} else {{
            btn.setAttribute("data-liked", "true");
            btn.style.color = "#4f46e5";
            btn.querySelector("svg").setAttribute("fill", "#4f46e5");
            btn.querySelector("svg").setAttribute("stroke", "#4f46e5");
            localStorage.setItem(msgId, "true");
        }}
    }}

    function retryResponse() {{
        if (window.parent) {{
            const parentUrl = new URL(window.parent.location.href);
            parentUrl.searchParams.set("retry", "{idx}");
            window.parent.location.href = parentUrl.toString();
        }}
    }}
    </script>
    <style>
    .message-actions {{
        display: flex;
        gap: 8px;
        align-items: center;
        margin-top: 4px;
        height: 100%;
    }}
    .action-btn {{
        background: none;
        border: none;
        cursor: pointer;
        padding: 6px;
        border-radius: 4px;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        box-sizing: border-box;
    }}
    .action-btn:hover {{
        color: #4f46e5;
    }}
    .icon-svg {{
        width: 16px;
        height: 16px;
    }}
    </style>
    """
    import streamlit.components.v1 as components
    components.html(action_btn_html, height=35)


# Initialize LLM Engine
if groq_api_key:
    st.session_state.llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="openai/gpt-oss-120b",
        temperature=0.2
    )
    st.session_state.graph = build_rag_graph()

# Render Chat Interface
if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-state">
            <div class="welcome-icon">📚</div>
            <div class="welcome-title">Welcome to your PDF RAG Workspace</div>
            <div class="welcome-subtitle">
                Upload PDF documents using the sidebar, then ask questions about their content.
                The AI will retrieve relevant context and cite sources.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_action_buttons(message["content"], idx)

if prompt_input := st.chat_input("Ask anything about your documents..."):
    if not groq_api_key:
        st.warning("Please provide a valid Groq API Key in the .env file to continue.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing context & formulating response..."):
                initial_state = {"question": prompt_input}
                output = st.session_state.graph.invoke(initial_state)
                response_text = output.get("response", "No answer could be generated.")

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                render_action_buttons(response_text, len(st.session_state.messages) - 1)