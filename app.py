"""
app.py — Class 8 Science AI Tutor
Streamlit single-page chat interface.

Run:
    pip install streamlit anthropic sentence-transformers faiss-cpu
    streamlit run app.py
"""

import os
import re
import json
import time
import numpy as np
import streamlit as st
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CORPUS_PATH  = "class8_science.jsonl"
INDEX_PATH   = "class8_science.faiss"
MODEL        = "llama-3.1-8b-instant"   # Free on Groq; swap to llama3-70b-8192 for best quality
TOP_K        = 5
MAX_TOKENS   = 512

SYSTEM_PROMPT = """\
You are an AI tutor for Class 8 students following the NCERT Science curriculum.

Rules you MUST follow:
1. Answer ONLY using the provided textbook context. Do NOT use outside knowledge.
2. Use simple, clear language appropriate for a 13-14 year old student.
3. Structure your answer in 2-4 short paragraphs. Use bullet points only for lists.
4. At the end of your answer, add a 'Sources:' section that lists the key phrases 
   from the context you drew upon (quote 8-12 words per source).
5. If the question cannot be answered from the context, respond with exactly:
   "I'm focused on Class 8 Science. This topic isn't in my textbook context — 
   try rephrasing or ask about a different chapter."
6. Never mention Claude, AI, or language models.
"""

# ─────────────────────────────────────────────
# Cached resource loading (runs once per session)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embeddings model…")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Loading corpus & FAISS index…")
def load_index_and_docs():
    docs = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(obj["text"])

    index = faiss.read_index(INDEX_PATH)
    return docs, index

def get_client() -> Groq:
    api_key = st.session_state.get("api_key") or os.environ.get("GROQ_API_KEY", "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")
    if not api_key:
        return None
    return Groq(api_key=api_key)

# ─────────────────────────────────────────────
# RAG functions
# ─────────────────────────────────────────────
def retrieve(query: str, docs, index, embedder, k: int = TOP_K):
    q_emb = embedder.encode([query]).astype(np.float32)
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, k)
    return [
        {"text": docs[idx], "score": float(scores[0][rank])}
        for rank, idx in enumerate(indices[0])
    ]


def build_context_block(chunks: list[dict]) -> str:
    block = ""
    for i, c in enumerate(chunks, 1):
        block += f"[Context {i} | relevance={c['score']:.2f}]\n{c['text']}\n\n"
    return block


def ask_tutor(
    query: str,
    history: list[dict],
    docs,
    index,
    embedder,
    client: Groq,
) -> dict:
    chunks = retrieve(query, docs, index, embedder)
    context_block = build_context_block(chunks)

    # Build messages — include last 4 history entries for continuity
    messages = history[-4:].copy()
    messages.append({
        "role": "user",
        "content": f"Textbook Context:\n{context_block}\nStudent Question: {query}"
    })

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages
    )

    answer = response.choices[0].message.content
    usage  = response.usage
    return {
        "answer": answer,
        "chunks": chunks,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
    }


# ─────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Class 8 Science AI Tutor",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .chat-user   { background: #e8f4fd; border-radius: 12px; padding: 10px 14px; margin: 4px 0; color: #000000; }
    .chat-tutor  { background: #f0fdf4; border-radius: 12px; padding: 10px 14px; margin: 4px 0; color: #000000; }
    .source-box  { background: #fffbe6; border-left: 3px solid #f0a500;
                   padding: 8px 12px; border-radius: 6px; font-size: 0.83em; margin-top: 6px; color: #000000; }
    .metric-chip { background: #e9ecef; border-radius: 20px;
                   padding: 2px 10px; font-size: 0.78em; display: inline-block; margin: 2px; color: #000000; }
    h1 { font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Simple_icon_book.svg/240px-Simple_icon_book.svg.png", width=60)
    st.title("🎓 AI Tutor Settings")

    api_key_input = st.text_input(
        "Groq API Key (free)",
        type="password",
        value=st.session_state.get("api_key", ""),
        placeholder="gsk_…",
        help="Get yours FREE at console.groq.com — no credit card needed"
    )
    if api_key_input:
        st.session_state["api_key"] = api_key_input

    st.divider()

    st.markdown("**📖 Corpus**")
    st.caption("NCERT Class 8 Science — 13 Chapters")

    st.markdown("**🤖 Model**")
    model_choice = st.selectbox(
        "LLM",
        ["llama-3.1-8b-instant", "llama3-70b-8192", "llama-3.3-70b-versatile"],
        index=0,
        help="All free on Groq. 8b = faster; 70b = higher quality"
    )
    MODEL = model_choice

    top_k = st.slider("Retrieval chunks (k)", 3, 8, TOP_K)

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.raw_history = []
        st.rerun()

    st.divider()
    st.caption("⚡ Built with sentence-transformers + FAISS + Llama 3 (Groq)")

# ─────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────
st.title("📚 Class 8 Science AI Tutor")
st.caption("Ask any question from your NCERT Science textbook. Answers are strictly from the book.")

# Suggested questions
with st.expander("💡 Sample questions to try", expanded=False):
    cols = st.columns(2)
    samples = [
        "What is friction and what causes it?",
        "Explain the cell theory.",
        "How does the greenhouse effect work?",
        "What is the difference between metals and non-metals?",
        "Describe the water cycle.",
        "What are microorganisms? Give examples.",
        "How does sound travel?",
        "What is adolescence?",
    ]
    for i, q in enumerate(samples):
        col = cols[i % 2]
        if col.button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state["prefill"] = q

# ─────────────────────────────────────────────
# State initialisation
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []       # Display list: {role, content, meta}
if "raw_history" not in st.session_state:
    st.session_state.raw_history = []    # Anthropic message format for context

# ─────────────────────────────────────────────
# Render chat history
# ─────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="chat-user">🧑‍🎓 <b>You:</b> {msg["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="chat-tutor">📖 <b>Tutor:</b><br>{msg["content"]}</div>',
            unsafe_allow_html=True
        )
        if msg.get("chunks"):
            with st.expander("📎 Source snippets from textbook", expanded=False):
                for i, chunk in enumerate(msg["chunks"][:3], 1):
                    st.markdown(
                        f'<div class="source-box"><b>Source {i}</b> '
                        f'(relevance: {chunk["score"]:.2f})<br>{chunk["text"][:300]}…</div>',
                        unsafe_allow_html=True
                    )
        if msg.get("meta"):
            m = msg["meta"]
            st.markdown(
                f'<span class="metric-chip">⬆️ {m["input_tokens"]} in</span>'
                f'<span class="metric-chip">⬇️ {m["output_tokens"]} out</span>',
                unsafe_allow_html=True
            )

# ─────────────────────────────────────────────
# Input handling
# ─────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input(
    "Ask a science question…",
    key="chat_input"
)
query = user_input or prefill

if query:
    # Validate API key
    client = get_client()
    if not client:
        st.error("⚠️ Please enter your Groq API key in the sidebar. Get one free at console.groq.com")
        st.stop()

    # Load resources (cached after first load)
    try:
        embedder = load_embedder()
        docs, idx = load_index_and_docs()
    except FileNotFoundError as e:
        st.error(
            f"❌ Required file missing: {e}\n\n"
            "Please run the Jupyter notebook first to generate "
            "`class8_science.jsonl` and `class8_science.faiss`."
        )
        st.stop()

    # Display user message
    st.markdown(
        f'<div class="chat-user">🧑‍🎓 <b>You:</b> {query}</div>',
        unsafe_allow_html=True
    )
    st.session_state.messages.append({"role": "user", "content": query})

    # Generate response
    with st.spinner("Searching textbook…"):
        try:
            result = ask_tutor(
                query,
                st.session_state.raw_history,
                docs, idx, embedder, client
            )
            answer  = result["answer"]
            chunks  = result["chunks"]
            meta    = {"input_tokens": result["input_tokens"],
                       "output_tokens": result["output_tokens"]}

            # Update raw history (clean, no context block)
            st.session_state.raw_history.append({"role": "user",      "content": query})
            st.session_state.raw_history.append({"role": "assistant", "content": answer})

            # Display answer
            st.markdown(
                f'<div class="chat-tutor">📖 <b>Tutor:</b><br>{answer}</div>',
                unsafe_allow_html=True
            )
            with st.expander("📎 Source snippets from textbook", expanded=False):
                for i, chunk in enumerate(chunks[:3], 1):
                    st.markdown(
                        f'<div class="source-box"><b>Source {i}</b> '
                        f'(relevance: {chunk["score"]:.2f})<br>{chunk["text"][:300]}…</div>',
                        unsafe_allow_html=True
                    )
            st.markdown(
                f'<span class="metric-chip">⬆️ {meta["input_tokens"]} in</span>'
                f'<span class="metric-chip">⬇️ {meta["output_tokens"]} out</span>',
                unsafe_allow_html=True
            )

            # Save to session
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "chunks": chunks,
                "meta": meta
            })

        except Exception as e:
            if "auth" in str(e).lower() or "api_key" in str(e).lower():
                st.error("❌ Invalid Groq API key. Get a free one at console.groq.com")
            else:
                st.error(f"❌ Something went wrong: {e}")
