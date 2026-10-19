import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import streamlit as st

from rag_pipeline import (
    answer_with_llm,
    answer_with_ollama,
    create_index,
    read_pdfs,
    retrieve,
)

st.set_page_config(page_title="ResearchIQ | Research Workspace", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).resolve().parent
DB = ROOT / "researchiq_history.db"
css_path = ROOT / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def database():
    return sqlite3.connect(DB, timeout=10)


def initialize_database():
    with database() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            documents TEXT NOT NULL,
            sources TEXT NOT NULL,
            mode TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")


def get_history():
    with database() as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM conversations ORDER BY id DESC")]


def save_history(question, answer, documents, sources, mode):
    with database() as conn:
        conn.execute(
            "INSERT INTO conversations (question, answer, documents, sources, mode, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (question, answer, json.dumps(documents), json.dumps(sources), mode, datetime.now().isoformat(timespec="seconds")),
        )


@st.cache_resource(show_spinner="Loading research search model...")
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


initialize_database()
if "chat" not in st.session_state:
    st.session_state.chat = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"

PAGES = ["Dashboard", "My Documents", "AI Research Chat", "History", "Sources", "Settings"]
NAV_LABELS = {
    "Dashboard": "⌂  Dashboard",
    "My Documents": "▤  My Documents",
    "AI Research Chat": "▣  AI Research Chat",
    "History": "◷  History",
    "Sources": "⌕  Sources",
    "Settings": "⚙  Settings",
}

with st.sidebar:
    st.markdown('<div class="nav-caption">WORKSPACE</div>', unsafe_allow_html=True)
    for page in PAGES:
        if st.button(NAV_LABELS[page], key=f"nav_{page}", use_container_width=True,
                     type="primary" if st.session_state.current_page == page else "secondary"):
            st.session_state.current_page = page
            st.rerun()
    st.divider()
    st.markdown('<div class="sidebar-model"><div class="sidebar-model-label">LOCAL AI MODEL</div><strong>Llama 3.2 · Ollama</strong><div class="model-status">● Runs on your Mac</div></div>', unsafe_allow_html=True)
    st.caption("Streamlit · FAISS · Ollama")

st.markdown("""<div class="hero"><h1>Research<span>IQ</span></h1><p>Your Intelligent Research Companion</p></div>""", unsafe_allow_html=True)

page = st.session_state.current_page

# The uploader is a main-content widget, never duplicated in the sidebar.
if page in ("Dashboard", "My Documents", "AI Research Chat"):
    st.markdown('<div class="section-heading">Upload Research Papers</div>', unsafe_allow_html=True)
    st.caption("Add one or more text-based PDFs. Files stay in your current local app session.")
    uploaded_files = st.file_uploader("Choose research PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed", key="research_pdfs")
else:
    uploaded_files = st.session_state.get("research_pdfs", []) or []

files = uploaded_files or []
documents = [f.name for f in files]
chunks = []
index = None
if files:
    signature = tuple((f.name, f.size, f.getvalue()[:100]) for f in files)
    if st.session_state.get("file_signature") != signature:
        with st.spinner("Extracting and indexing your research papers..."):
            new_chunks = read_pdfs(files)
            if not new_chunks:
                st.error("No readable text was found. Try a text-based PDF.")
                st.stop()
            new_index = create_index(new_chunks, get_embedder())
            st.session_state.chunks = new_chunks
            st.session_state.index = new_index
            st.session_state.file_signature = signature
            st.session_state.chat = []
            st.session_state.last_sources = []
    chunks = st.session_state.chunks
    index = st.session_state.index


def show_chat_history():
    for entry in st.session_state.chat:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])


def answer_question(question, mode, api_key):
    sources = retrieve(question, chunks, index, get_embedder())
    topic_terms = ["topic", "title", "main idea", "overview", "what is this paper about"]
    if any(term in question.lower() for term in topic_terms):
        first_page = [p for p in chunks if p["page"] == 1]
        if first_page:
            sources = first_page[:3]

    source_records = [
        {"file": p["file"], "page": int(p["page"]), "text": p["text"]}
        for p in sources[:3]
    ]

    if mode == "Local AI Answer (Free)":
        try:
            answer = answer_with_ollama(question, sources)
        except Exception as exc:
            answer = f"Local AI could not respond. Ensure Ollama is running. Details: {exc}"
    elif mode == "AI-generated answer (API key)":
        try:
            secret_key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            secret_key = ""
        key = api_key or os.getenv("OPENAI_API_KEY", "") or secret_key
        if not key:
            answer = "Enter an API key in Settings to use the cloud AI mode."
        else:
            try:
                answer = answer_with_llm(question, sources, key)
            except Exception as exc:
                answer = f"Cloud AI could not respond: {exc}"
    else:
        answer = "### Retrieved evidence\n\n"
        for n, p in enumerate(source_records, 1):
            answer += f"**Source {n}: {p['file']} — Page {p['page']}**\n\n{p['text'][:450]}\n\n"
        answer += "*These are retrieved excerpts, not a generated summary.*"

    if source_records and mode != "Source passages (Free)":
        answer += "\n\n---\n**Retrieved sources (verify against the PDF):**\n\n"
        for n, p in enumerate(source_records, 1):
            answer += f"{n}. {p['file']} — Page {p['page']}\n\n"

    st.session_state.chat.append({"question": question, "answer": answer})
    st.session_state.last_sources = source_records
    save_history(question, answer, documents, source_records, mode)


if page == "Dashboard":
    history = get_history()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Research Papers", len(files))
    c2.metric("Text Chunks", len(chunks))
    c3.metric("Saved Questions", len(history))
    c4.metric("Local Model", "Llama 3.2")
    st.markdown('<div class="section-heading">Recent Research Activity</div>', unsafe_allow_html=True)
    if history:
        for item in history[:5]:
            st.markdown(f"**{item['question']}**")
            st.caption(item["created_at"])
    else:
        st.info("Your research questions will appear here after you start chatting.")
    st.info("Select **AI Research Chat** on the left to ask questions about your PDFs.")

elif page == "My Documents":
    st.markdown('<div class="section-heading">My Documents</div>', unsafe_allow_html=True)
    if not files:
        st.info("Upload your research PDFs above.")
    for f in files:
        with st.container(border=True):
            st.markdown(f"**📄 {f.name}**")
            st.caption(f"{sum(p['file'] == f.name for p in chunks)} indexed text chunks · {f.size / 1024 / 1024:.2f} MB")

elif page == "AI Research Chat":
    if not files:
        st.info("Upload a PDF above to start chatting with your research assistant.")
    show_chat_history()
    question = st.chat_input("Ask a question about your research papers...", disabled=not bool(files))
    if question and files:
        with st.spinner("ResearchIQ is analyzing your papers..."):
            answer_question(question, st.session_state.get("answer_mode", "Local AI Answer (Free)"), st.session_state.get("api_key", ""))
        st.rerun()

elif page == "History":
    st.markdown('<div class="section-heading">Research History</div>', unsafe_allow_html=True)
    history = get_history()
    query = st.text_input("Search saved questions and answers", placeholder="Search your history...")
    filtered = [h for h in history if query.lower() in (h["question"] + " " + h["answer"] + " " + h["documents"]).lower()]
    st.caption(f"{len(filtered)} saved conversation(s)")
    if not filtered:
        st.info("No matching conversations found.")
    for h in filtered:
        with st.expander(f"{h['question']} · {h['created_at']}"):
            st.markdown(h["answer"])
            st.caption(f"Mode: {h['mode']} · Documents: {', '.join(json.loads(h['documents']))}")
            try:
                saved_sources = json.loads(h["sources"])
            except (ValueError, TypeError):
                saved_sources = []
            for source in saved_sources:
                st.caption(f"{source['file']} — Page {source['page']}")
    if history:
        st.divider()
        confirm = st.checkbox("I understand this permanently deletes saved conversation history.")
        if st.button("Clear All History", disabled=not confirm):
            with database() as conn:
                conn.execute("DELETE FROM conversations")
            st.rerun()

elif page == "Sources":
    st.markdown('<div class="section-heading">Retrieved Sources</div>', unsafe_allow_html=True)
    st.caption("Evidence passages from your most recent question in this session. Check the original PDF to verify claims.")
    if not st.session_state.last_sources:
        st.info("Ask a question in AI Research Chat to see retrieved passages here.")
    for n, source in enumerate(st.session_state.last_sources, 1):
        with st.expander(f"Source {n} · {source['file']} · Page {source['page']}"):
            st.write(source["text"])

elif page == "Settings":
    st.markdown('<div class="section-heading">Settings</div>', unsafe_allow_html=True)
    st.markdown("**AI Model Settings**")
    st.caption("Current local model: Llama 3.2 (3B), running through Ollama")
    st.radio("Answer mode", ["Local AI Answer (Free)", "Source passages (Free)", "AI-generated answer (API key)"], key="answer_mode")
    if st.session_state.answer_mode == "AI-generated answer (API key)":
        st.text_input("OpenAI API key", type="password", key="api_key")
    st.divider()
    st.markdown("**Research Engine**")
    st.write("Embeddings: all-MiniLM-L6-v2 · Search: FAISS · PDF extraction: PyMuPDF")
    st.divider()
    st.markdown("**Storage & Privacy**")
    st.write("Conversation history is saved locally in researchiq_history.db. Uploaded PDFs and their index are session-based.")
    st.caption("Before publishing online, add user authentication and private storage for each account.")
