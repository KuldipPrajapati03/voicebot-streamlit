import os
import uuid
import ast
import shutil
import base64
import mimetypes
import json

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv


load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")

# Normal RAG model
RAG_MODEL = "openai/gpt-oss-120b"

# Vision model
# Change this if your Groq account uses a different supported vision model.
VISION_MODEL = "qwen/qwen3.6-27b"

STORE_DIR = "faiss_index_store_new"


# ==========================================
# 1. Streamlit Theme Configuration
# ==========================================

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
# 3. Custom CSS
# ==========================================

chat_css = """
<style>

/* ============================================
   GLOBAL TYPOGRAPHY & THEME
   ============================================ */

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

    --shadow-md:
        0 4px 6px -1px rgb(0 0 0 / 0.1),
        0 2px 4px -2px rgb(0 0 0 / 0.1);

    --shadow-lg:
        0 10px 15px -3px rgb(0 0 0 / 0.1),
        0 4px 6px -4px rgb(0 0 0 / 0.1);

    --shadow-xl:
        0 20px 25px -5px rgb(0 0 0 / 0.1),
        0 8px 10px -6px rgb(0 0 0 / 0.1);

    --radius-sm: 6px;
    --radius-md: 8px;
    --radius-lg: 12px;
    --radius-xl: 16px;
}


/* ============================================
   DARK THEME
   ============================================ */

[data-theme="dark"] {

    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-tertiary: #1e293b;
    --bg-sidebar: #0f172a;
    --bg-card: #1e293b;
    --bg-input: #1e293b;
    --bg-hover: #334155;

    --bg-chat-user:
        linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);

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

    --shadow-sm:
        0 1px 2px 0 rgb(0 0 0 / 0.3);

    --shadow-md:
        0 4px 6px -1px rgb(0 0 0 / 0.4),
        0 2px 4px -2px rgb(0 0 0 / 0.4);

    --shadow-lg:
        0 10px 15px -3px rgb(0 0 0 / 0.4),
        0 4px 6px -4px rgb(0 0 0 / 0.4);

    --shadow-xl:
        0 20px 25px -5px rgb(0 0 0 / 0.5),
        0 8px 10px -6px rgb(0 0 0 / 0.5);
}


/* ============================================
   GLOBAL
   ============================================ */

html,
body,
[class*="css"] {
    font-family:
        'Inter',
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Roboto,
        Helvetica,
        Arial,
        sans-serif !important;
}

.stApp {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}


/* ============================================
   HEADER
   ============================================ */

header[data-testid="stHeader"] {

    background:
        rgba(255, 255, 255, 0.85) !important;

    backdrop-filter: blur(12px) !important;

    -webkit-backdrop-filter:
        blur(12px) !important;

    border-bottom:
        1px solid var(--border-primary) !important;

    height: 3.75rem !important;
}

[data-theme="dark"]
header[data-testid="stHeader"] {

    background:
        rgba(15, 23, 42, 0.85) !important;
}

header[data-testid="stHeader"]::before {

    content:
        "⚡ Enterprise RAG Assistant";

    position: absolute;

    left: 1.5rem;

    top: 50%;

    transform:
        translateY(-50%);

    font-size: 0.95rem;

    font-weight: 600;

    color:
        var(--text-primary) !important;
}


/* ============================================
   SIDEBAR
   ============================================ */

[data-testid="stSidebar"] {

    background-color:
        var(--bg-sidebar) !important;

    border-right:
        1px solid var(--border-primary) !important;
}

[data-testid="stSidebar"] hr {

    border-color:
        var(--border-primary) !important;

    margin:
        1.25rem 0 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {

    color:
        var(--text-secondary) !important;

    font-size:
        0.85rem !important;

    text-transform:
        uppercase !important;

    letter-spacing:
        0.05em !important;

    font-weight:
        600 !important;
}


/* ============================================
   DEVELOPER INFO
   ============================================ */

.developer-info {
    margin-bottom: 8px;
}

.developer-label {

    font-size: 11px;

    text-transform: uppercase;

    letter-spacing: 0.05em;

    color:
        var(--text-muted);

    font-weight: 600;
}

.developer-name {

    font-size: 14px;

    font-weight: 600;

    color:
        var(--text-primary);

    margin-top: 2px;
}

.developer-contact {

    font-size: 12px;

    color:
        var(--accent-primary);

    text-decoration: none;
}


/* ============================================
   INPUTS
   ============================================ */

div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] > div > div {

    background-color:
        var(--bg-input) !important;

    border:
        1px solid var(--border-primary) !important;

    color:
        var(--text-primary) !important;

    border-radius:
        var(--radius-md) !important;
}

div[data-testid="stTextInput"] input:focus {

    border-color:
        var(--border-focus) !important;

    box-shadow:
        0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

div[data-testid="stWidgetLabel"] p {

    color:
        var(--text-secondary) !important;

    font-size:
        0.825rem !important;

    font-weight:
        500 !important;
}


/* ============================================
   FILE UPLOADER
   ============================================ */

div[data-testid="stFileUploader"] {

    background-color:
        var(--bg-input) !important;

    border:
        2px dashed var(--border-secondary) !important;

    border-radius:
        var(--radius-lg) !important;

    padding:
        10px !important;
}

div[data-testid="stFileUploader"]:hover {

    border-color:
        var(--accent-primary) !important;

    background-color:
        rgba(99, 102, 241, 0.05) !important;
}


/* ============================================
   BUTTONS
   ============================================ */

div.stButton > button {

    border-radius:
        var(--radius-md) !important;

    font-weight:
        500 !important;

    font-size:
        0.875rem !important;

    border:
        1px solid var(--border-primary) !important;

    background-color:
        var(--bg-input) !important;

    color:
        var(--text-primary) !important;

    transition:
        all 0.2s ease-in-out !important;
}

div.stButton > button:hover {

    background-color:
        var(--bg-hover) !important;

    border-color:
        var(--accent-primary) !important;
}

div.stButton > button[kind="primary"] {

    background:
        linear-gradient(
            135deg,
            #4f46e5 0%,
            #6366f1 100%
        ) !important;

    border: none !important;

    color:
        var(--text-inverse) !important;

    box-shadow:
        var(--shadow-sm) !important;
}

div.stButton > button[kind="primary"]:hover {

    background:
        linear-gradient(
            135deg,
            #4338ca 0%,
            #4f46e5 100%
        ) !important;

    box-shadow:
        var(--shadow-md) !important;
}


/* ============================================
   CHAT
   ============================================ */

/* Chat text */
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] span {
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}


/* ============================================
   BASE CHAT MESSAGE
   ============================================ */

div[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 10px 0 !important;

    /* Important: allow children to move left/right */
    width: 100% !important;
}


/* Message content wrapper */
div[data-testid="stChatMessage"] > div:nth-child(2) {
    width: fit-content !important;
    max-width: 82% !important;
    flex-grow: 0 !important;
}


/* ============================================
   USER MESSAGE → RIGHT SIDE
   ============================================ */

div[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarUser"]
) {
    flex-direction: row-reverse !important;
    justify-content: flex-start !important;
}


div[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarUser"]
) > div:nth-child(2) {
    margin-left: auto !important;
    margin-right: 12px !important;
}


/* User bubble */
div[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarUser"]
) [data-testid="stChatMessageContent"] {

    background: var(--bg-chat-user) !important;

    color: var(--text-inverse) !important;

    border: none !important;

    border-radius: 16px 16px 4px 16px !important;

    padding: 12px 18px !important;

    box-shadow: var(--shadow-sm) !important;
}


/* ============================================
   ASSISTANT / BOT MESSAGE → LEFT SIDE
   ============================================ */

div[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) {
    flex-direction: row !important;
    justify-content: flex-start !important;
}


div[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) > div:nth-child(2) {
    margin-left: 12px !important;
    margin-right: auto !important;
}


/* Assistant bubble */
div[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) [data-testid="stChatMessageContent"] {

    background-color: var(--bg-chat-assistant) !important;

    border: 1px solid var(--border-primary) !important;

    color: var(--text-primary) !important;

    border-radius: 16px 16px 16px 4px !important;

    padding: 14px 20px !important;

    box-shadow: var(--shadow-sm) !important;
}


/* ============================================
   PREVENT CONTENT FROM STRETCHING
   ============================================ */

div[data-testid="stChatMessageContent"] {
    width: fit-content !important;
    max-width: 100% !important;
}


/* ============================================
   USER TEXT COLOR
   ============================================ */

div[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarUser"]
) [data-testid="stChatMessageContent"] p,
div[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarUser"]
) [data-testid="stChatMessageContent"] span,
div[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarUser"]
) [data-testid="stChatMessageContent"] li {
    color: var(--text-inverse) !important;
}


/* ============================================
   BOT TEXT COLOR
   ============================================ */

div[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) [data-testid="stChatMessageContent"] p,
div[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) [data-testid="stChatMessageContent"] span,
div[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) [data-testid="stChatMessageContent"] li {
    color: var(--text-primary) !important;
}

/* ============================================
   CHAT INPUT
   ============================================ */

div[data-testid="stChatInput"] {

    background-color:
        var(--bg-secondary) !important;

    border:
        1px solid var(--border-primary) !important;

    border-radius:
        var(--radius-lg) !important;

    box-shadow:
        var(--shadow-md) !important;
}

div[data-testid="stChatInput"]:focus-within {

    border-color:
        var(--border-focus) !important;

    box-shadow:
        0 0 0 3px
        rgba(99, 102, 241, 0.1),
        var(--shadow-md) !important;
}

div[data-testid="stChatInput"] textarea {

    color:
        var(--text-primary) !important;

    font-size:
        0.95rem !important;
}


/* ============================================
   IMAGE ATTACHMENT
   ============================================ */

.image-attachment-box {

    border:
        1px solid var(--border-primary);

    background:
        var(--bg-card);

    border-radius:
        12px;

    padding:
        12px;

    margin-bottom:
        12px;

    box-shadow:
        var(--shadow-sm);
}

.image-attachment-title {

    color:
        var(--text-primary);

    font-size:
        14px;

    font-weight:
        600;

    margin-bottom:
        6px;
}


/* ============================================
   CONFIG POPOVER / RIGHT-SIDE DRAWER
   ============================================ */

/* Put only the Config trigger in the top-right header area.
   IMPORTANT: do NOT make the popover body fixed. Streamlit renders
   the popover body in a portal and positions it automatically. */
div[data-testid="stPopover"] {
    position: fixed !important;
    top: 0.72rem !important;
    right: 1.25rem !important;
    z-index: 1000000 !important;
    width: 88px !important;
}

button[data-testid="stPopoverButton"] {
    width: 88px !important;
    height: 34px !important;
    min-height: 34px !important;
    padding: 0 12px !important;
    border-radius: 8px !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
}

/* The popover body is rendered in a portal. Let Streamlit/BaseWeb
   calculate its position, and only control its size/appearance. */
div[data-testid="stPopoverBody"] {
    z-index: 1000000 !important;
    width: 320px !important;
    max-width: calc(100vw - 24px) !important;
    max-height: calc(100vh - 5.25rem) !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    box-sizing: border-box !important;
    background-color: var(--bg-sidebar) !important;
    border: 1px solid var(--border-primary) !important;
    border-radius: 12px !important;
    box-shadow: var(--shadow-xl) !important;
    padding: 1rem !important;
}

/* Compatibility with older Streamlit DOM. */
div[data-testid="stPopoverPanel"] {
    z-index: 1000000 !important;
    width: 320px !important;
    max-width: calc(100vw - 24px) !important;
    max-height: calc(100vh - 5.25rem) !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    box-sizing: border-box !important;
    background-color: var(--bg-sidebar) !important;
    border: 1px solid var(--border-primary) !important;
    border-radius: 12px !important;
    box-shadow: var(--shadow-xl) !important;
}

/* Keep the configuration content compact. */
div[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"],
div[data-testid="stPopoverPanel"] [data-testid="stVerticalBlock"] {
    width: 100% !important;
}

/* Remove unnecessary vertical gaps inside Config. */
div[data-testid="stPopoverBody"] [data-testid="stElementContainer"],
div[data-testid="stPopoverPanel"] [data-testid="stElementContainer"] {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

div[data-testid="stPopoverBody"] .developer-info,
div[data-testid="stPopoverPanel"] .developer-info {
    margin: 0 !important;
    padding: 0 !important;
}

div[data-testid="stPopoverBody"] .developer-label,
div[data-testid="stPopoverPanel"] .developer-label {
    font-size: 10px !important;
    line-height: 1.2 !important;
    margin: 0 !important;
}

div[data-testid="stPopoverBody"] .developer-name,
div[data-testid="stPopoverPanel"] .developer-name {
    display: inline-block !important;
    font-size: 13px !important;
    line-height: 1.25 !important;
    margin: 2px 0 0 !important;
}

/* Compact divider between Developer and Knowledge Base. */
div[data-testid="stPopoverBody"] .config-divider,
div[data-testid="stPopoverPanel"] .config-divider {
    height: 1px !important;
    margin: 10px 0 12px !important;
    background: var(--border-primary) !important;
}

/* Make Knowledge Base small enough to stay on one line. */
div[data-testid="stPopoverBody"] h1,
div[data-testid="stPopoverBody"] h2,
div[data-testid="stPopoverBody"] h3,
div[data-testid="stPopoverPanel"] h1,
div[data-testid="stPopoverPanel"] h2,
div[data-testid="stPopoverPanel"] h3 {
    font-size: 1.05rem !important;
    line-height: 1.25 !important;
    margin: 0 0 10px !important;
}

/* Compact labels and uploader text. */
div[data-testid="stPopoverBody"] label,
div[data-testid="stPopoverPanel"] label {
    font-size: 0.82rem !important;
}

div[data-testid="stPopoverBody"] [data-testid="stFileUploaderDropzone"],
div[data-testid="stPopoverPanel"] [data-testid="stFileUploaderDropzone"] {
    padding: 0.65rem !important;
}

/* ============================================
   PERSISTENT CONFIG DRAWER
   ============================================ */

/* The Config trigger lives in its own marked Streamlit block so it can
   stay fixed in the top-right without affecting page layout. */
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .config-trigger-marker) {
    position: fixed !important;
    top: -0.28rem !important;
    right: 1.25rem !important;
    width: 88px !important;
    z-index: 1000000 !important;
    margin: 0 !important;
    padding: 0 !important;
}

.config-trigger-marker {
    display: none !important;
}

/* Style ONLY the actual Config trigger. */
div[data-testid="stVerticalBlock"]:has(.config-trigger-marker) button {
    width: 88px !important;
    min-width: 88px !important;
    max-width: 88px !important;
    height: 34px !important;
    min-height: 34px !important;
    max-height: 34px !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    background: #e2e8f0 !important;
    color: #334155 !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    line-height: 1 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

div[data-testid="stVerticalBlock"]:has(.config-trigger-marker) button:hover {
    background: #dbe4ee !important;
    border-color: #94a3b8 !important;
}

div[data-testid="stVerticalBlock"]:has(.config-trigger-marker) button p {
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    text-align: center !important;
    white-space: nowrap !important;
    line-height: 1 !important;
}

/* Persistent drawer. Unlike st.popover, this remains mounted after a
   dialog rerun, so closing PDF Chunks Viewer does NOT close Config. */
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .config-drawer-marker) {
    position: fixed !important;
    top: 4.5rem !important;
    right: 1.25rem !important;
    width: 320px !important;
    max-width: calc(100vw - 24px) !important;
    max-height: calc(100vh - 5.25rem) !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    box-sizing: border-box !important;
    padding: 14px 16px 16px !important;
    background: var(--bg-sidebar) !important;
    border: 1px solid var(--border-primary) !important;
    border-radius: 12px !important;
    box-shadow: var(--shadow-xl) !important;
    z-index: 1000000 !important;
    margin: 0 !important;
}

.config-drawer-marker {
    display: none !important;
}

/* Keep drawer content compact. */
div[data-testid="stVerticalBlock"]:has(.config-drawer-marker) [data-testid="stElementContainer"] {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

div[data-testid="stVerticalBlock"]:has(.config-drawer-marker) .developer-info {
    margin: 0 !important;
    padding: 2px 2px 4px !important;
}

div[data-testid="stVerticalBlock"]:has(.config-drawer-marker) .developer-label {
    display: block !important;
    font-size: 10px !important;
    line-height: 1.25 !important;
    margin: 0 0 4px !important;
    padding: 0 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

div[data-testid="stVerticalBlock"]:has(.config-drawer-marker) .developer-name {
    display: inline-block !important;
    font-size: 13px !important;
    line-height: 1.3 !important;
    font-weight: 600 !important;
    margin: 0 !important;
    padding: 0 !important;
}

div[data-testid="stVerticalBlock"]:has(.config-drawer-marker) .config-divider {
    height: 1px !important;
    margin: 10px 0 12px !important;
    background: var(--border-primary) !important;
}

div[data-testid="stVerticalBlock"]:has(.config-drawer-marker) h1,
div[data-testid="stVerticalBlock"]:has(.config-drawer-marker) h2,
div[data-testid="stVerticalBlock"]:has(.config-drawer-marker) h3 {
    font-size: 1.05rem !important;
    line-height: 1.3 !important;
    margin: 0 0 12px !important;
    padding: 0 !important;
}

/* Keep Process / Clear Store buttons the SAME size whether or not
   documents are present. Do not let the button width shrink based on
   the current content/state of the drawer. */
div[data-testid="stVerticalBlock"]:has(.config-drawer-marker) [data-testid="stHorizontalBlock"] {
    width: 100% !important;
    display: flex !important;
    gap: 10px !important;
    align-items: stretch !important;
    box-sizing: border-box !important;
}

div[data-testid="stVerticalBlock"]:has(.config-drawer-marker) [data-testid="stHorizontalBlock"] > [data-testid="column"] {
    flex: 1 1 0% !important;
    width: 0 !important;
    min-width: 0 !important;
    max-width: none !important;
    box-sizing: border-box !important;
}

div[data-testid="stVerticalBlock"]:has(.config-drawer-marker) [data-testid="stHorizontalBlock"] > [data-testid="column"] > div {
    width: 100% !important;
    box-sizing: border-box !important;
}

div[data-testid="stVerticalBlock"]:has(.config-drawer-marker) [data-testid="stHorizontalBlock"] div.stButton,
div[data-testid="stVerticalBlock"]:has(.config-drawer-marker) [data-testid="stHorizontalBlock"] div.stButton > button {
    width: 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;
    min-height: 38px !important;
    height: 38px !important;
    max-height: 38px !important;
    padding: 0 10px !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    line-height: 1 !important;
    white-space: nowrap !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

div[data-testid="stVerticalBlock"]:has(.config-drawer-marker) [data-testid="stHorizontalBlock"] button p {
    width: 100% !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    margin: 0 !important;
    padding: 0 !important;
    text-align: center !important;
}

/* PDF buttons in Active Context. Target only the buttons marked
   config-pdf-button-marker so Streamlit's generic button rules cannot
   distort the document button. */
div[data-testid="stVerticalBlock"]:has(.config-pdf-button-marker) {
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

div[data-testid="stVerticalBlock"]:has(.config-pdf-button-marker) .config-pdf-button-marker {
    display: none !important;
}

div[data-testid="stVerticalBlock"]:has(.config-pdf-button-marker) div.stButton {
    width: 100% !important;
    margin: 0 !important;
}

div[data-testid="stVerticalBlock"]:has(.config-pdf-button-marker) div.stButton > button {
    width: 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;
    height: 40px !important;
    min-height: 40px !important;
    max-height: 40px !important;
    padding: 0 12px !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    border-radius: 8px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
}

div[data-testid="stVerticalBlock"]:has(.config-pdf-button-marker) div.stButton > button p {
    display: block !important;
    width: 100% !important;
    min-width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    text-align: center !important;
    line-height: 1.2 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

div[data-testid="stVerticalBlock"]:has(.config-drawer-marker) label {
    font-size: 0.82rem !important;
}

div[data-testid="stVerticalBlock"]:has(.config-drawer-marker) [data-testid="stFileUploaderDropzone"] {
    padding: 0.65rem !important;
}

/* ============================================
   DIALOG
   ============================================ */

/* ============================================
   MODAL LAYERING
   ============================================ */

/* The PDF Chunks Viewer is a true modal and must ALWAYS sit above
   the Config popover, including its backdrop and close button. */
div[data-testid="stDialog"] {
    z-index: 2000000 !important;
}

/* Streamlit may put the dialog inside an additional wrapper. Raise
   the wrapper as well without changing its positioning behavior. */
div[data-testid="stDialog"] > div,
div[data-testid="stDialog"] div[role="dialog"] {
    z-index: 2000001 !important;
}

/* Do not force position on the dialog. Streamlit controls the modal
   positioning; we only style its visual appearance. */
div[role="dialog"] {
    background-color:
        var(--bg-card) !important;

    border:
        1px solid var(--border-primary) !important;

    border-radius:
        var(--radius-xl) !important;

    box-shadow:
        var(--shadow-xl) !important;
}

/* If Streamlit exposes the modal overlay separately, keep it above
   the Config popover too. */
div[data-testid="stDialog"]::before {
    z-index: 2000000 !important;
}

div[data-testid*="DialogHeader"] {

    display:
        none !important;
}

div[role="dialog"] button[kind="header"] svg {

    color:
        var(--text-primary) !important;
}


/* ============================================
   WELCOME
   ============================================ */

.welcome-state {

    display:
        flex;

    flex-direction:
        column;

    align-items:
        center;

    justify-content:
        center;

    height:
        50vh;

    text-align:
        center;

    color:
        var(--text-muted);

    margin-top:
        5vh;
}

.welcome-icon {

    font-size:
        48px;

    margin-bottom:
        16px;
}

.welcome-title {

    font-size:
        20px;

    font-weight:
        600;

    color:
        var(--text-primary);

    margin-bottom:
        8px;
}

.welcome-subtitle {

    font-size:
        14px;

    max-width:
        400px;

    line-height:
        1.6;

    color:
        var(--text-secondary);
}


/* ============================================
   ALERTS
   ============================================ */

[data-testid="stAlert"] {

    background-color:
        var(--bg-card) !important;

    border:
        1px solid var(--border-primary) !important;

    color:
        var(--text-primary) !important;

    border-radius:
        var(--radius-md) !important;
}

[data-testid="stAlert"] p {

    color:
        var(--text-primary) !important;
}


/* ============================================
   EXPANDER
   ============================================ */

div[data-testid="stExpander"] {

    background-color:
        var(--bg-card) !important;

    border:
        1px solid var(--border-primary) !important;

    border-radius:
        var(--radius-md) !important;
}

div[data-testid="stExpander"] summary {

    color:
        var(--text-primary) !important;
}


/* ============================================
   TEXT AREA
   ============================================ */

textarea {

    background-color:
        var(--bg-input) !important;

    color:
        var(--text-primary) !important;

    border-color:
        var(--border-primary) !important;
}

textarea:disabled {

    -webkit-text-fill-color:
        var(--text-primary) !important;

    opacity:
        1 !important;
}


/* ============================================
   SELECTBOX
   ============================================ */

div[data-testid="stSelectbox"] span {

    color:
        var(--text-primary) !important;
}


/* ============================================
   POPOVER
   ============================================ */

div[data-baseweb="popover"] ul,
div[data-baseweb="menu"],
div[data-baseweb="popover"] div {

    background-color:
        var(--bg-card) !important;
}

div[data-baseweb="popover"] li,
div[data-baseweb="menu"] li {

    color:
        var(--text-primary) !important;
}

div[data-baseweb="popover"] li:hover,
div[data-baseweb="menu"] li:hover {

    background-color:
        var(--bg-hover) !important;
}


/* ============================================
   GENERAL
   ============================================ */

.stApp p,
.stApp span,
.stApp label {

    color:
        var(--text-primary);
}

.stCaption,
[data-testid="stCaptionContainer"] {

    color:
        var(--text-muted) !important;
}

.stSpinner > div {

    color:
        var(--text-secondary) !important;
}


/* ============================================
   HIDE STREAMLIT MENU
   ============================================ */

#MainMenu {

    visibility:
        hidden !important;
}

[data-testid="stToolbar"] {

    visibility:
        hidden !important;
}

</style>
"""

st.html(chat_css)


# ==========================================
# 4. Session State Initialization
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "graph" not in st.session_state:
    st.session_state.graph = None

# if "uploaded_image" not in st.session_state:
#     st.session_state.uploaded_image = None
#
# if "image_question_mode" not in st.session_state:
#     st.session_state.image_question_mode = False


# ==========================================
# 5. Text Extraction & Vectorstore
# ==========================================

def get_pdf_documents(file):

    """
    Extract PDF elements and convert them into
    LangChain Documents with metadata.
    """

    temp_file = f"temp_{uuid.uuid4()}.pdf"

    with open(temp_file, "wb") as f:
        f.write(file.getvalue())

    try:

        with st.spinner(
            f"Extracting text from {file.name}..."
        ):

            elements = partition_pdf(
                filename=temp_file,
                infer_table_structure=True,
                chunking_strategy="by_title",
                max_characters=4000,
                new_after_n_chars=3800,
                combine_text_under_n_chars=2000,
            )

    finally:

        if os.path.exists(temp_file):
            os.remove(temp_file)

    with st.spinner(
        f"Chunking {file.name}..."
    ):

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        all_chunks = []

        for element in elements:

            page_num = (
                element.metadata.page_number
                if hasattr(element, "metadata")
                and element.metadata.page_number
                else 1
            )

            element_id = getattr(
                element,
                "id",
                str(uuid.uuid4())
            )

            element_text = getattr(
                element,
                "text",
                ""
            )

            if not element_text:
                continue

            splits = text_splitter.split_text(
                element_text
            )

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


def get_vectorstore(
    text_chunks,
    store_name,
    embeddings
):

    with st.spinner(
        "Saving chunks to vectorstore & generating embeddings..."
    ):

        if st.session_state.vectorstore is not None:

            vectorstore = (
                st.session_state.vectorstore
            )

            vectorstore.add_documents(
                text_chunks
            )

        elif os.path.exists(store_name):

            vectorstore = FAISS.load_local(
                store_name,
                embeddings,
                allow_dangerous_deserialization=True
            )

            vectorstore.add_documents(
                text_chunks
            )

        else:

            vectorstore = FAISS.from_documents(
                text_chunks,
                embeddings
            )

        vectorstore.save_local(
            store_name
        )

    return vectorstore


# ==========================================
# 6. Conversation Memory
# ==========================================

def get_last_three_conversations():

    """
    Return the last 3 completed
    user -> assistant conversation pairs.

    The current unanswered question is ignored.
    """

    messages = st.session_state.get(
        "messages",
        []
    )

    conversations = []

    i = 0

    while i < len(messages) - 1:

        current = messages[i]
        next_message = messages[i + 1]

        if (
            current.get("role") == "user"
            and next_message.get("role") == "assistant"
        ):

            conversations.append(
                {
                    "question":
                        current.get("content", ""),

                    "answer":
                        next_message.get("content", "")
                }
            )

            i += 2

        else:

            i += 1

    # Only last 3 conversations
    conversations = conversations[-3:]

    if not conversations:
        return "No previous conversation."

    history_parts = []

    for idx, conversation in enumerate(
        conversations,
        start=1
    ):

        history_parts.append(
            f"""
Conversation {idx}

User:
{conversation["question"]}

Assistant:
{conversation["answer"]}
"""
        )

    return "\n".join(history_parts)


# ==========================================
# 7. LangGraph State
# ==========================================

class State(TypedDict):

    question: str
    context_with_ids: str
    retrieved_docs: List[Document]
    chat_history: str
    response: str


# ==========================================
# 8. Retrieve Node
# ==========================================

def retrieve_node(state: State):

    """
    Retrieve relevant PDF documents and
    include last 3 conversation pairs.
    """

    chat_history = (
        get_last_three_conversations()
    )

    vectorstore = (
        st.session_state.get(
            "vectorstore"
        )
    )

    if vectorstore is None:

        return {
            "context_with_ids": "",
            "retrieved_docs": [],
            "chat_history": chat_history
        }

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4
        }
    )

    docs = retriever.invoke(
        state["question"]
    )

    context_with_ids = "\n\n".join(
        [
            (
                f"ID: "
                f"{doc.metadata.get('element_id', 'N/A')}\n"
                f"Content: "
                f"{doc.page_content}"
            )
            for doc in docs
        ]
    )

    return {

        "context_with_ids":
            context_with_ids,

        "retrieved_docs":
            docs,

        "chat_history":
            chat_history
    }


# ==========================================
# 9. Generate Node
# ==========================================

def generate_node(state: State):

    """
    Generate response using:

    - PDF context
    - Last 3 conversations
    - Current question
    """

    prompt_template = (
        ChatPromptTemplate.from_template(
            """
You are an AI assistant designed to answer
questions using the provided document context
and recent conversation history.

========================================
RECENT CONVERSATION
========================================

{chat_history}


========================================
DOCUMENT CONTEXT
========================================

{context_with_ids}


========================================
CURRENT QUESTION
========================================

{question}


========================================
INSTRUCTIONS
========================================

1. Use the recent conversation history to
understand follow-up questions.

2. If the current question contains words
such as:

- it
- they
- them
- this
- that
- previous
- above
- earlier
- same
- more
- what about it

use the previous conversation to understand
what the user is referring to.

3. If the document context contains relevant
information, prioritize the document context.

4. Do not invent facts from the document.

5. If the document context does not contain
relevant information, you may use general
knowledge.

6. Answer naturally and clearly.

7. Do not unnecessarily repeat the previous
conversation.

8. After the answer, add exactly:

Source: ['element_id']

when document context was used.

If document context was not used, add:

Source: None


Answer:
"""
        )
    )

    chain = (
        prompt_template
        | st.session_state.llm
    )

    res = chain.invoke(
        {
            "context_with_ids":
                state.get(
                    "context_with_ids",
                    ""
                ),

            "chat_history":
                state.get(
                    "chat_history",
                    ""
                ),

            "question":
                state["question"]
        }
    )

    result_text = res.content

    if "\nSource:" in result_text:

        answer_parts = result_text.rsplit(
            "\nSource:",
            1
        )

        answer = (
            answer_parts[0]
            .strip()
        )

        source_type = (
            answer_parts[1]
            .strip()
        )

    else:

        answer = (
            result_text
            .strip()
        )

        source_type = "None"

    # ======================================
    # Map Element IDs -> Source + Pages
    # ======================================

    sources_pages = {}

    if (
        source_type.startswith("[")
        and source_type.endswith("]")
    ):

        try:

            source_element_ids = (
                ast.literal_eval(
                    source_type
                )
            )

            for doc in state.get(
                "retrieved_docs",
                []
            ):

                element_id = (
                    doc.metadata.get(
                        "element_id"
                    )
                )

                if (
                    element_id
                    in source_element_ids
                ):

                    src = (
                        doc.metadata.get(
                            "source",
                            "Unknown Document"
                        )
                    )

                    page = (
                        doc.metadata.get(
                            "page",
                            "N/A"
                        )
                    )

                    if src not in sources_pages:

                        sources_pages[src] = set()

                    sources_pages[src].add(
                        page
                    )

        except (
            ValueError,
            SyntaxError,
            TypeError
        ):

            sources_pages = {}

    # ======================================
    # Format Sources
    # ======================================

    if sources_pages:

        citation_strings = []

        for src, pages in sources_pages.items():

            sorted_pages = sorted(
                list(pages),
                key=lambda x: (
                    int(x)
                    if str(x).isdigit()
                    else str(x)
                )
            )

            pages_str = ", ".join(
                map(
                    str,
                    sorted_pages
                )
            )

            citation_strings.append(
                f"**{src}** "
                f"(Page {pages_str})"
            )

        formatted_sources = (
            " | ".join(
                citation_strings
            )
        )

        final_response = (
            f"{answer}\n\n"
            f"**Sources:** "
            f"{formatted_sources}"
        )

    else:

        final_response = (
            f"{answer}\n\n"
            f"*(AI knowledge)*"
        )

    return {
        "response":
            final_response
    }


# ==========================================
# 10. Build RAG Graph
# ==========================================

def build_rag_graph():

    graph_builder = (
        StateGraph(State)
    )

    graph_builder.add_node(
        "retrieve",
        retrieve_node
    )

    graph_builder.add_node(
        "generate",
        generate_node
    )

    graph_builder.add_edge(
        START,
        "retrieve"
    )

    graph_builder.add_edge(
        "retrieve",
        "generate"
    )

    graph_builder.add_edge(
        "generate",
        END
    )

    return graph_builder.compile()


# ==========================================
# 11. PDF Chunk Viewer
# ==========================================

@st.dialog(
    "PDF Chunks Viewer",
    width="large"
)
def show_pdf_chunks(pdf_name):

    vectorstore = (
        st.session_state.vectorstore
    )

    if vectorstore is None:

        st.write(
            "Vectorstore is empty."
        )

        return

    doc_dict = (
        vectorstore
        .docstore
        ._dict
    )

    chunks_info = []

    for doc_id, doc in doc_dict.items():

        src = doc.metadata.get(
            "source"
        )

        if src == pdf_name:

            page = doc.metadata.get(
                "page",
                1
            )

            chunks_info.append(
                (
                    page,
                    doc.page_content
                )
            )

    chunks_info.sort(
        key=lambda x: x[0]
    )

    col1, col2, col3 = (
        st.columns(
            [5, 2, 2]
        )
    )

    with col1:

        st.markdown(
            f"### Chunk Details · {len(chunks_info)} chunks for `{pdf_name}`"
        )

    with col2:

        expand_all = st.checkbox(
            "Expand All",
            value=False,
            key="expand_all_chunks_checkbox"
        )

    with col3:

        all_text = "\n\n".join(
            [
                text
                for _, text in chunks_info
            ]
        )

        escaped_text = json.dumps(
            all_text
        )

        copy_button_html = f"""
        <button
            id="copyBtn"
            onclick="copyToClip()"
            style="
                border-radius:8px;
                padding:8px 12px;
                border:1px solid #e2e8f0;
                background:#f8fafc;
                cursor:pointer;
            "
        >
            📋 Copy Text
        </button>

        <script>

        function copyToClip() {{

            const text =
                {escaped_text};

            navigator.clipboard
                .writeText(text)
                .then(() => {{

                    const btn =
                        document.getElementById(
                            "copyBtn"
                        );

                    btn.innerHTML =
                        "✓ Copied!";

                    setTimeout(() => {{

                        btn.innerHTML =
                            "📋 Copy Text";

                    }}, 2000);

                }});

        }}

        </script>
        """

        components.html(
            copy_button_html,
            height=45
        )

    st.markdown(
        "<div style='margin-bottom:12px;'></div>",
        unsafe_allow_html=True
    )

    with st.container(height=450):

        for idx, (page, text) in enumerate(
            chunks_info
        ):

            with st.expander(
                f"📄 Chunk {idx + 1} "
                f"(Page {page})",
                expanded=expand_all
            ):

                st.text_area(
                    label=f"Chunk {idx + 1} Content",
                    value=text,
                    height=150,
                    disabled=True,
                    key=f"dialog_chunk_{idx}",
                    label_visibility="collapsed"
                )


# ==========================================
# 12. Image Question Function
# ==========================================

def ask_image_question(
    image_file,
    question
):

    """
    Send image + question + recent conversation
    to a vision-capable Groq model.
    """

    image_bytes = (
        image_file.getvalue()
    )

    base64_image = (
        base64.b64encode(
            image_bytes
        ).decode("utf-8")
    )

    mime_type = (
        image_file.type
        or mimetypes.guess_type(
            image_file.name
        )[0]
        or "image/png"
    )

    image_url = (
        f"data:{mime_type};base64,"
        f"{base64_image}"
    )

    # Get last 3 completed conversations
    chat_history = (
        get_last_three_conversations()
    )

    vision_llm = ChatGroq(
        groq_api_key=groq_api_key,

        model_name=VISION_MODEL,

        temperature=0.2
    )

    system_instruction = """
You are an AI vision assistant.

Analyze the provided image carefully.

The user may ask questions about:
- text
- charts
- tables
- graphs
- screenshots
- diagrams
- documents
- objects
- people
- visual relationships

Use the recent conversation history to
understand follow-up questions.

If the user says:
- "it"
- "this"
- "that"
- "the above"
- "the chart"
- "the image"
- "what about this"

use the previous conversation to determine
what they mean.

Do not claim to see something that is not
actually visible in the image.

If text is visible in the image, read it
carefully.

Answer clearly and directly.
"""

    user_text = f"""
Recent conversation:

{chat_history}

Current question:

{question}
"""

    response = vision_llm.invoke(
        [
            {
                "role": "system",
                "content":
                    system_instruction
            },

            {
                "role": "user",

                "content": [

                    {
                        "type": "text",
                        "text": user_text
                    },

                    {
                        "type": "image_url",

                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ]
    )

    return response.content


# ==========================================
# 13. Action Buttons
# ==========================================

def get_plain_text(content):

    if "\n\n**Sources:**" in content:

        return (
            content
            .split(
                "\n\n**Sources:**"
            )[0]
            .strip()
        )

    if "\n\n*(AI knowledge)*" in content:

        return (
            content
            .split(
                "\n\n*(AI knowledge)*"
            )[0]
            .strip()
        )

    return content.strip()


def render_action_buttons(
    content,
    idx
):

    plain_text = get_plain_text(
        content
    )

    escaped_plain_text = (
        json.dumps(
            plain_text
        )
    )

    action_btn_html = f"""

    <div class="message-actions">

        <button
            id="copyBtn"
            class="action-btn"
            title="Copy response"
            onclick="copyText()"
        >

            <svg
                class="icon-svg"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
            >

                <rect
                    x="9"
                    y="9"
                    width="13"
                    height="13"
                    rx="2"
                ></rect>

                <path
                    d="M5 15H4a2 2 0
                       0 1-2-2V4a2 2
                       0 1 2-2h9a2 2
                       0 1 2 1v1"
                ></path>

            </svg>

        </button>


        <button
            id="likeBtn"
            class="action-btn"
            title="Like response"
            onclick="toggleLike()"
            data-liked="false"
        >

            <svg
                class="icon-svg"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
            >

                <path
                    d="M14 9V5a3 3 0
                       0 0-3-3l-4 9v11h11.28
                       a2 2 0 0 0 2-1.7l1.38-9
                       a2 2 0 0 0-2-2.3z
                       M7 22H4a2 2 0
                       0 1-2-2v-7a2 2
                       0 0 1 2-2h3"
                ></path>

            </svg>

        </button>

    </div>


    <script>

    const msgId =
        "like_state_{idx}";

    const textToCopy =
        {escaped_plain_text};


    function copyText() {{

        navigator.clipboard
            .writeText(textToCopy)
            .then(function() {{

                const btn =
                    document.getElementById(
                        "copyBtn"
                    );

                btn.innerHTML = "✓";

                setTimeout(function() {{

                    btn.innerHTML = `
                        <svg
                            class="icon-svg"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            stroke-width="2"
                        >
                            <rect
                                x="9"
                                y="9"
                                width="13"
                                height="13"
                                rx="2"
                            ></rect>

                            <path
                                d="M5 15H4a2 2
                                   0 0 1-2-2V4a2 2
                                   0 0 1 2-2h9a2 2
                                   0 0 1 2 1v1"
                            ></path>
                        </svg>
                    `;

                }}, 2000);

            }});

    }}


    function toggleLike() {{

        const btn =
            document.getElementById(
                "likeBtn"
            );

        const liked =
            btn.getAttribute(
                "data-liked"
            ) === "true";

        if (liked) {{

            btn.setAttribute(
                "data-liked",
                "false"
            );

            btn.style.color =
                "#94a3b8";

            btn.querySelector(
                "svg"
            ).removeAttribute(
                "fill"
            );

            localStorage.setItem(
                msgId,
                "false"
            );

        }} else {{

            btn.setAttribute(
                "data-liked",
                "true"
            );

            btn.style.color =
                "#4f46e5";

            btn.querySelector(
                "svg"
            ).setAttribute(
                "fill",
                "#4f46e5"
            );

            localStorage.setItem(
                msgId,
                "true"
            );

        }}

    }}

    </script>


    <style>

    .message-actions {{

        display:
            flex;

        gap:
            8px;

        align-items:
            center;

        margin-top:
            4px;
    }}

    .action-btn {{

        background:
            none;

        border:
            none;

        cursor:
            pointer;

        padding:
            6px;

        border-radius:
            4px;

        display:
            flex;

        align-items:
            center;

        justify-content:
            center;

        color:
            #94a3b8;
    }}

    .action-btn:hover {{

        background:
            #f1f5f9;

        color:
            #4f46e5;
    }}

    .icon-svg {{

        width:
            16px;

        height:
            16px;
    }}

    </style>
    """

    components.html(
        action_btn_html,
        height=35
    )


# ==========================================
# 14. Configuration Popup
# ==========================================

processed_files = []

if st.session_state.get("vectorstore") is not None:
    try:
        sources = set()
        for doc in (
            st.session_state
            .vectorstore
            .docstore
            ._dict
            .values()
        ):
            src = doc.metadata.get("source")
            if src:
                sources.add(src)

        processed_files = sorted(list(sources))

    except Exception:
        processed_files = []


# Keep Config open across reruns. This is important when the PDF Chunks
# dialog is opened from inside Config: closing the dialog causes a rerun,
# so a native st.popover would otherwise close automatically.
if "config_open" not in st.session_state:
    st.session_state.config_open = False


def toggle_config():
    st.session_state.config_open = not st.session_state.config_open


# Config trigger button.
with st.container():
    st.html('<div class="config-trigger-marker"></div>')

    config_label = "Config"

    st.button(
        config_label,
        key="config_toggle_btn",
        on_click=toggle_config,
        type="secondary",
    )


# Persistent right-side configuration drawer.
if st.session_state.config_open:
    with st.container():
        st.html('<div class="config-drawer-marker"></div>')

        # ======================================
        # Developer Information
        # ======================================
        st.html("""
        <div class="developer-info">
            <div class="developer-label">Developed By</div>
            <a href="https://www.linkedin.com/in/kuldip-prajapati"
               target="_blank"
               class="developer-name">
               Kuldip Prajapati
            </a>
        </div>
        """)

        st.html('<div class="config-divider"></div>')

        # ======================================
        # Knowledge Base
        # ======================================
        st.header("📂 Knowledge Base")

        uploaded_file = st.file_uploader(
            "Upload Document",
            type=["pdf"],
            accept_multiple_files=True,
            key="config_pdf_uploader"
        )

        col1, col2 = st.columns(2)

        with col1:
            process_btn = st.button(
                "Process",
                use_container_width=True,
                type="primary",
                key="config_process_btn"
            )

        with col2:
            clear_btn = st.button(
                "Clear Store",
                use_container_width=True,
                key="config_clear_btn"
            )

        # ======================================
        # Process PDFs
        # ======================================
        if process_btn:
            if not uploaded_file:
                st.warning("Please upload at least one PDF document.")
            else:
                chunks = []

                for file in uploaded_file:
                    file_chunks = get_pdf_documents(file)
                    chunks.extend(file_chunks)

                if chunks:
                    vectorstore = get_vectorstore(
                        chunks,
                        STORE_DIR,
                        st.session_state.embeddings
                    )

                    st.session_state.vectorstore = vectorstore

                    st.success("Documents indexed successfully!")
                    st.rerun()
                else:
                    st.warning(
                        "No content could be extracted from the uploaded PDF(s)."
                    )

        # ======================================
        # Clear Vectorstore
        # ======================================
        if clear_btn:
            if os.path.exists(STORE_DIR):
                shutil.rmtree(
                    STORE_DIR,
                    ignore_errors=True
                )

            st.session_state.vectorstore = None
            st.session_state.messages = []

            st.success("Vectorstore cleared.")
            st.rerun()

        # ======================================
        # Processed PDFs / Active Context
        # ======================================
        if processed_files:
            st.divider()
            st.header("📄 Active Context")
            st.caption("Click a file to inspect parsed chunks:")

            for i, pdf_name in enumerate(processed_files):
                # Give each PDF button its own marker so the CSS targets
                # ONLY the document buttons and not Process/Clear/Delete.
                with st.container():
                    st.html('<div class="config-pdf-button-marker"></div>')

                    if st.button(
                        f"📄 {pdf_name}",
                        key=f"config_view_pdf_{i}",
                        use_container_width=True
                    ):
                        # Keep Config open while the modal is displayed and
                        # after the modal is closed.
                        st.session_state.config_open = True
                        show_pdf_chunks(pdf_name)

            st.write("")

            pdf_to_delete = st.selectbox(
                "Manage Documents:",
                ["-- Select file to remove --"] + processed_files,
                key="config_pdf_to_delete_selectbox"
            )

            if pdf_to_delete != "-- Select file to remove --":
                if st.button(
                    "Delete Document",
                    use_container_width=True,
                    type="primary",
                    key="config_delete_document_btn"
                ):
                    vectorstore = st.session_state.vectorstore
                    doc_ids_to_delete = []

                    for doc_id, doc in list(
                        vectorstore.docstore._dict.items()
                    ):
                        if doc.metadata.get("source") == pdf_to_delete:
                            doc_ids_to_delete.append(doc_id)

                    if doc_ids_to_delete:
                        total_docs = len(vectorstore.docstore._dict)

                        if len(doc_ids_to_delete) >= total_docs:
                            if os.path.exists(STORE_DIR):
                                shutil.rmtree(
                                    STORE_DIR,
                                    ignore_errors=True
                                )

                            st.session_state.vectorstore = None
                            st.session_state.messages = []

                        else:
                            vectorstore.delete(doc_ids_to_delete)
                            vectorstore.save_local(STORE_DIR)

                    st.success(f"Removed: {pdf_to_delete}")
                    st.rerun()


# ==========================================
# 15. Initialize Embeddings
# ==========================================

if st.session_state.embeddings is None:

    with st.spinner(
        "Initializing embedding engine..."
    ):

        st.session_state.embeddings = (
            HuggingFaceEmbeddings(
                model_name=
                    "all-MiniLM-L6-v2"
            )
        )


# ==========================================
# 16. Load Existing Vectorstore
# ==========================================

if (
    st.session_state.vectorstore
    is None
):

    if os.path.exists(
        STORE_DIR
    ):

        try:

            st.session_state.vectorstore = (
                FAISS.load_local(
                    STORE_DIR,
                    st.session_state.embeddings,
                    allow_dangerous_deserialization=True
                )
            )

        except Exception:

            st.session_state.vectorstore = None


# ==========================================
# 19. Initialize LLM
# ==========================================

if groq_api_key:

    st.session_state.llm = ChatGroq(

        groq_api_key=
            groq_api_key,

        model_name=
            RAG_MODEL,

        temperature=
            0.2
    )

    st.session_state.graph = (
        build_rag_graph()
    )


# ==========================================
# 20. Retry Handling
# ==========================================

try:

    q_params = st.query_params

    if "retry" in q_params:

        retry_idx = int(
            q_params["retry"]
        )

        del st.query_params["retry"]

        if (
            "messages"
            in st.session_state
            and
            retry_idx
            <
            len(
                st.session_state.messages
            )
        ):

            user_msg_idx = (
                retry_idx - 1
            )

            if (
                user_msg_idx >= 0
                and
                user_msg_idx
                <
                len(
                    st.session_state.messages
                )
            ):

                question = (
                    st.session_state
                    .messages[
                        user_msg_idx
                    ]["content"]
                )

                if groq_api_key:

                    output = (
                        st.session_state
                        .graph
                        .invoke(
                            {
                                "question":
                                    question
                            }
                        )
                    )

                    response_text = (
                        output.get(
                            "response",
                            "No answer could be generated."
                        )
                    )

                    st.session_state.messages[
                        retry_idx
                    ]["content"] = (
                        response_text
                    )

                    st.rerun()

except Exception:

    pass


# ==========================================
# 21. Chat History Rendering
# ==========================================

if not st.session_state.messages:

    st.html("""
    <div class="welcome-state">
        <div class="welcome-icon">📚</div>
        <div class="welcome-title">Welcome to your PDF RAG Workspace</div>
        <div class="welcome-subtitle">
            Upload PDF documents using the sidebar, then ask questions about their content.
            The AI will retrieve relevant context and cite sources.
            <br><br>
            You can also click <b>+</b> to upload an image and ask questions about it.
        </div>
    </div>
    """)


else:

    for idx, message in enumerate(
        st.session_state.messages
    ):

        with st.chat_message(
            message["role"]
        ):

            # Image attached to a user message
            if (
                message["role"]
                == "user"
                and
                message.get(
                    "image_data"
                )
            ):

                try:

                    image_bytes = base64.b64decode(
                        message["image_data"]
                    )

                    st.image(
                        image_bytes,
                        width=350
                    )

                except Exception:

                    pass

            st.markdown(
                message["content"]
            )

            if (
                message["role"]
                == "assistant"
            ):

                render_action_buttons(
                    message["content"],
                    idx
                )


# ==========================================
# 22. Image Attachment UI
# ==========================================

# plus_col, status_col = st.columns(
#     [0.08, 0.92]
# )
#
# with plus_col:
#
#     if st.button(
#         "+",
#         key="image_plus_button",
#         help="Attach an image"
#     ):
#
#         st.session_state.image_question_mode = (
#             not st.session_state.image_question_mode
#         )
#
#         st.rerun()
#
#
# with status_col:
#
#     if st.session_state.uploaded_image:
#
#         st.markdown(
#             f"""
#             <div
#                 style="
#                     padding:8px 12px;
#                     border:1px solid #cbd5e1;
#                     border-radius:8px;
#                     background:#f8fafc;
#                     font-size:13px;
#                 "
#             >
#                 🖼️ Attached:
#                 <b>
#                     {st.session_state.uploaded_image.name}
#                 </b>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )


# ==========================================
# 23. Image Uploader
# ==========================================

# if st.session_state.image_question_mode:
#
#     st.markdown(
#         """
#         <div class="image-attachment-box">
#
#             <div class="image-attachment-title">
#                 🖼️ Attach an image
#             </div>
#
#         </div>
#         """,
#         unsafe_allow_html=True
#     )
#
#     image_file = st.file_uploader(
#         "Choose an image",
#         type=[
#             "png",
#             "jpg",
#             "jpeg",
#             "webp"
#         ],
#         key="chat_image_uploader",
#         label_visibility="collapsed"
#     )
#
#     if image_file is not None:
#
#         st.session_state.uploaded_image = (
#             image_file
#         )
#
#         st.image(
#             image_file,
#             caption="Attached image",
#             width=350
#         )
#
#         if st.button(
#             "Remove Image",
#             key="remove_chat_image"
#         ):
#
#             st.session_state.uploaded_image = None
#
#             st.session_state.image_question_mode = False
#
#             st.rerun()


# ==========================================
# 24. Chat Input
# ==========================================
# ==========================================
# 24. Chat Input + Image Upload
# ==========================================

chat_input = st.chat_input(
    "Ask anything about your documents or image...",
    accept_file=True,
    file_type=[
        "png",
        "jpg",
        "jpeg",
        "webp"
    ],
    max_upload_size=200
)



# ==========================================
# 25. Handle User Question
# ==========================================

# ==========================================
# 25. Handle User Question
# ==========================================

if chat_input:

    prompt_input = chat_input.text.strip()

    uploaded_files = chat_input.files

    image_file = None

    # --------------------------------------
    # Get uploaded image
    # --------------------------------------

    if uploaded_files:

        image_file = uploaded_files[0]

    # --------------------------------------
    # Check if anything was submitted
    # --------------------------------------

    if not prompt_input and image_file is None:

        st.warning(
            "Please enter a question or attach an image."
        )

    elif not groq_api_key:

        st.warning(
            "Please provide a valid Groq API Key "
            "in the .env file to continue."
        )

    else:

        # ==================================
        # IMAGE QUESTION
        # ==================================

        if image_file is not None:

            # ----------------------------------
            # Convert image to base64
            # ----------------------------------

            image_base64 = (
                base64.b64encode(
                    image_file.getvalue()
                ).decode("utf-8")
            )

            # ----------------------------------
            # If no question was entered,
            # give a default instruction
            # ----------------------------------

            if not prompt_input:

                prompt_input = (
                    "Please analyze this image "
                    "and describe what you can see."
                )

            # ----------------------------------
            # Save user message
            # ----------------------------------

            st.session_state.messages.append(
                {
                    "role": "user",

                    "content": prompt_input,

                    "image_data": image_base64
                }
            )

            # ----------------------------------
            # Display user message
            # ----------------------------------

            with st.chat_message("user"):

                st.image(
                    image_file,
                    width=350
                )

                st.markdown(
                    prompt_input
                )

            # ----------------------------------
            # Analyze image
            # ----------------------------------

            with st.chat_message("assistant"):

                with st.spinner(
                    "Analyzing image..."
                ):

                    try:

                        response_text = (
                            ask_image_question(
                                image_file,
                                prompt_input
                            )
                        )

                    except Exception as e:

                        response_text = (
                            "Sorry, I couldn't "
                            "analyze the image.\n\n"
                            f"Error: {str(e)}"
                        )

                    st.markdown(
                        response_text
                    )

                    # ----------------------------------
                    # Save assistant response
                    # ----------------------------------

                    st.session_state.messages.append(
                        {
                            "role":
                                "assistant",

                            "content":
                                response_text
                        }
                    )

                    render_action_buttons(
                        response_text,
                        len(
                            st.session_state.messages
                        ) - 1
                    )

            st.rerun()

        # ==================================
        # NORMAL PDF / RAG QUESTION
        # ==================================

        else:

            # ----------------------------------
            # Save user message
            # ----------------------------------

            st.session_state.messages.append(
                {
                    "role":
                        "user",

                    "content":
                        prompt_input
                }
            )

            # ----------------------------------
            # Display user message
            # ----------------------------------

            with st.chat_message("user"):

                st.markdown(
                    prompt_input
                )

            # ----------------------------------
            # Generate RAG answer
            # ----------------------------------

            with st.chat_message("assistant"):

                with st.spinner(
                    "Analyzing context & formulating response..."
                ):

                    initial_state = {
                        "question":
                            prompt_input
                    }

                    output = (
                        st.session_state
                        .graph
                        .invoke(
                            initial_state
                        )
                    )

                    response_text = (
                        output.get(
                            "response",
                            "No answer could be generated."
                        )
                    )

                    st.markdown(
                        response_text
                    )

                    # ----------------------------------
                    # Save assistant response
                    # ----------------------------------

                    st.session_state.messages.append(
                        {
                            "role":
                                "assistant",

                            "content":
                                response_text
                        }
                    )

                    render_action_buttons(
                        response_text,
                        len(
                            st.session_state.messages
                        ) - 1
                    )
