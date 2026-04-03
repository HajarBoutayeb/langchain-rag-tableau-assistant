import streamlit as st
import os
import time
import json
import tempfile
from datetime import datetime
from dotenv import load_dotenv

# LangChain Core & Models
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Loaders, Splitters & Memory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Advanced Retrieval (Multi-Query + Reranking) — langchain_classic for v1.2+
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

# Chains
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain

# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="Tableau AI Expert",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 2. CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    .stApp {
        background-color: #f8f9fb;
        color: #1a1a2e;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] * { color: #1a1a2e !important; }

    /* ── Cards ── */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    .source-card {
        background: #f0f4ff;
        border-left: 3px solid #4361ee;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 0.82em;
        color: #475569;
    }

    /* ── Status badges ── */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78em;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .status-ready   { background: #dcfce7; color: #15803d; }
    .status-waiting { background: #fef3c7; color: #b45309; }

    /* ── Chat bubbles ── */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 8px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
        border: 1px solid #e2e8f0;
        background: #ffffff;
        color: #1a1a2e;
    }
    .stButton > button:hover {
        background: #4361ee;
        color: #ffffff;
        border-color: #4361ee;
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px;
    }

    /* ── Hide branding ── */
    #MainMenu, footer { visibility: hidden; }

    /* ── Title area ── */
    .app-header {
        padding: 10px 0 20px 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. SESSION STATE DEFAULTS
# ─────────────────────────────────────────────
defaults = {
    "retriever": None,
    "rag_chain": None,
    "doc_stats": {},
    "show_sources": True,
    "debug_mode": False,
    "response_times": [],
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────
# 4. MODEL LOADING (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load embedding model and LLM once, reuse across reruns."""
    embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    if not GROQ_API_KEY:
        st.error("⚠️ GROQ_API_KEY not found in environment. Add it to your .env file.")
        st.stop()
    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        max_tokens=1024,
    )
    return embedding, llm

# ─────────────────────────────────────────────
# 5. RAG PIPELINE BUILDER
# ─────────────────────────────────────────────
def build_rag_pipeline(pdf_path: str, embedding_model, llm_model):
    """Load PDF → split → embed → multi-query → rerank → RAG chain."""

    # Phase 1 — Load & Split
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Phase 2 — Vector Store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )

    # Phase 3 — Multi-Query + Reranking
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_model,
    )
    compressor = FlashrankRerank(top_n=5)
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=mq_retriever,
    )

    # Phase 4 — History-Aware RAG Chain
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the conversation history and latest user question, "
         "rewrite it as a fully standalone question (no pronouns referencing history). "
         "Do NOT answer — only rewrite."
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm_model, final_retriever, contextualize_prompt
    )

    qa_system_prompt = (
        "You are an expert Tableau instructor and data visualization specialist. "
        "Answer ONLY using the provided context. "
        "If the context does not contain enough information, say so clearly — do not hallucinate. "
        "Be concise, structured, and use bullet points where appropriate.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    doc_chain = create_stuff_documents_chain(llm_model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, doc_chain)

    stats = {
        "pages": len(documents),
        "chunks": len(chunks),
        "built_at": datetime.now().strftime("%H:%M:%S"),
    }
    return final_retriever, rag_chain, stats


# ─────────────────────────────────────────────
# 6. SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Tableau AI Expert")
    st.markdown("---")

    # Status badge
    is_ready = st.session_state.retriever is not None
    badge_cls = "status-ready" if is_ready else "status-waiting"
    badge_txt = "● READY" if is_ready else "○ NO DOCUMENT"
    st.markdown(
        f'<span class="status-badge {badge_cls}">{badge_txt}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Upload
    st.markdown("### 📂 Document")
    uploaded_file = st.file_uploader(
        "Upload Tableau Course (PDF)",
        type="pdf",
        help="Upload a Tableau training PDF to power the knowledge base.",
    )

    if uploaded_file:
        st.info(f"📄 **{uploaded_file.name}** — {uploaded_file.size / 1024:.1f} KB")

        if st.button("🚀 Build Knowledge Base", use_container_width=True):
            with st.spinner("Processing document… this may take a minute."):
                try:
                    embedding_model, llm_model = load_models()

                    # Save to temp file (safer than hardcoded path)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    retriever, chain, stats = build_rag_pipeline(
                        tmp_path, embedding_model, llm_model
                    )
                    os.unlink(tmp_path)  # cleanup

                    st.session_state.retriever = retriever
                    st.session_state.rag_chain = chain
                    st.session_state.doc_stats = stats

                    st.success("✅ Knowledge base ready!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error building knowledge base:\n\n`{e}`")

    # Document Stats
    if st.session_state.doc_stats:
        st.markdown("### 📈 Document Stats")
        s = st.session_state.doc_stats
        col1, col2 = st.columns(2)
        col1.metric("Pages", s.get("pages", "—"))
        col2.metric("Chunks", s.get("chunks", "—"))
        st.caption(f"Built at {s.get('built_at', '—')}")

    st.markdown("---")

    # Settings
    st.markdown("### ⚙️ Settings")
    st.session_state.show_sources = st.toggle(
        "Show source excerpts", value=st.session_state.show_sources
    )
    st.session_state.debug_mode = st.toggle(
        "Debug mode", value=st.session_state.debug_mode
    )

    # Response time stats
    if st.session_state.response_times:
        avg_rt = sum(st.session_state.response_times) / len(st.session_state.response_times)
        st.markdown("---")
        st.caption(f"⏱ Avg response time: **{avg_rt:.1f}s** over {len(st.session_state.response_times)} queries")

    st.markdown("---")

    # Controls
    st.markdown("### 🛠️ Controls")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            msgs_key = st.session_state.get("chat_history", [])
            if "chat_history" in st.session_state:
                st.session_state["chat_history"] = []
            st.rerun()

    with col_b:
        # Export conversation
        msgs_obj = StreamlitChatMessageHistory(key="chat_history")
        if msgs_obj.messages:
            export_data = [
                {"role": m.type, "content": m.content}
                for m in msgs_obj.messages
            ]
            st.download_button(
                "💾 Export",
                data=json.dumps(export_data, ensure_ascii=False, indent=2),
                file_name=f"tableau_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True,
            )

# ─────────────────────────────────────────────
# 7. MAIN AREA
# ─────────────────────────────────────────────
st.markdown(
    '<div class="app-header">'
    '<h1 style="margin:0">🛡️ Tableau Expert System</h1>'
    '<p style="color:#6b7280;margin:4px 0 0 0">RAG · Multi-Query · Reranking · Conversation Memory</p>'
    '</div>',
    unsafe_allow_html=True,
)

# Suggested questions (only shown when system is ready and chat is empty)
msgs = StreamlitChatMessageHistory(key="chat_history")

if is_ready and not msgs.messages:
    st.markdown("#### 💡 Suggested questions")
    suggestions = [
        "What are the main chart types available in Tableau?",
        "How do I create a calculated field?",
        "What is the difference between a dimension and a measure?",
        "How can I connect Tableau to a SQL database?",
    ]
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        if cols[i % 2].button(suggestion, key=f"sug_{i}", use_container_width=True):
            # Inject as user message
            msgs.add_user_message(suggestion)
            st.rerun()

# Render chat history
for msg in msgs.messages:
    with st.chat_message(msg.type):
        st.write(msg.content)

# ─────────────────────────────────────────────
# 8. CHAT INPUT
# ─────────────────────────────────────────────
user_input = st.chat_input(
    "Ask anything about Tableau…",
    disabled=not is_ready,
)

if user_input:
    msgs.add_user_message(user_input)
    with st.chat_message("human"):
        st.write(user_input)

    if not is_ready:
        with st.chat_message("ai"):
            st.warning("⚠️ Please upload and process a PDF document first.")
    else:
        with st.chat_message("ai"):
            with st.spinner("Searching knowledge base…"):
                t0 = time.time()
                try:
                    response = st.session_state.rag_chain.invoke({
                        "input": user_input,
                        "chat_history": msgs.messages[:-1],  # exclude latest user msg
                    })
                    elapsed = time.time() - t0
                    st.session_state.response_times.append(elapsed)

                    answer = response.get("answer", "No answer returned.")
                    source_docs = response.get("context", [])

                    # Main answer
                    st.write(answer)

                    # Metadata row
                    st.caption(f"⏱ {elapsed:.1f}s · 📚 {len(source_docs)} sources used")

                    # Source excerpts
                    if st.session_state.show_sources and source_docs:
                        with st.expander(f"📄 Source excerpts ({len(source_docs)})"):
                            for i, doc in enumerate(source_docs, 1):
                                page = doc.metadata.get("page", "?")
                                source = doc.metadata.get("source", "document")
                                snippet = doc.page_content[:300].replace("\n", " ") + "…"
                                st.markdown(
                                    f'<div class="source-card">'
                                    f'<strong>#{i} — Page {page}</strong><br>{snippet}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                    # Debug
                    if st.session_state.debug_mode:
                        with st.expander("🔬 Debug info"):
                            st.json({
                                "num_sources": len(source_docs),
                                "response_time_s": round(elapsed, 3),
                                "history_length": len(msgs.messages),
                            })

                    msgs.add_ai_message(answer)

                except Exception as e:
                    error_msg = f"❌ An error occurred: {e}"
                    st.error(error_msg)
                    if st.session_state.debug_mode:
                        st.exception(e)
                    msgs.add_ai_message(error_msg)

# ─────────────────────────────────────────────
# 9. EMPTY STATE
# ─────────────────────────────────────────────
if not is_ready and not uploaded_file:
    st.markdown("<br>" * 3, unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center; color:#6b7280; padding: 60px 20px;">
            <div style="font-size:4em">📊</div>
            <h3 style="color:#374151">Upload a Tableau course PDF to get started</h3>
            <p style="color:#6b7280">The system will build a RAG knowledge base using<br>
            <strong>Multi-Query Retrieval</strong> + <strong>FlashRank Reranking</strong> + <strong>Conversation Memory</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )