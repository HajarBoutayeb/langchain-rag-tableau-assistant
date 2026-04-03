"""
eval.py — RAG Evaluation Dashboard
────────────────────────────────────
Standalone Streamlit page that evaluates the RAG pipeline
built in app.py using RAGAS metrics.

Run:
    streamlit run eval.py
Make sure the knowledge base was already built in app.py
(the retriever & rag_chain are stored in st.session_state).
"""

import os
import time
import json
import traceback
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# ── Models ────────────────────────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# ── RAG pipeline (same as app.py) ────────────────────────────────────────────
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain

# ── RAGAS ────────────────────────────────────────────────────────────────────
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="🧪",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
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

    /* ── Score cards ── */
    .score-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px 16px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        transition: box-shadow 0.2s;
    }
    .score-card:hover { box-shadow: 0 4px 16px rgba(67,97,238,0.12); }

    /* ── Score values ── */
    .score-value {
        font-size: 2.6em;
        font-weight: 700;
        margin: 10px 0 4px 0;
    }
    .score-good { color: #16a34a; }
    .score-mid  { color: #d97706; }
    .score-bad  { color: #dc2626; }

    /* ── Question chips ── */
    .question-chip {
        background: #f0f4ff;
        border: 1px solid #c7d2fe;
        border-radius: 8px;
        padding: 8px 14px;
        margin: 4px 0;
        font-size: 0.88em;
        color: #3730a3;
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
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

    /* ── Download buttons ── */
    [data-testid="stDownloadButton"] > button {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        color: #1a1a2e;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    [data-testid="stDownloadButton"] > button:hover {
        background: #4361ee;
        color: #ffffff;
        border-color: #4361ee;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Progress bar ── */
    [data-testid="stProgress"] > div > div {
        background: #4361ee;
    }

    /* ── Dividers ── */
    hr { border-color: #e2e8f0 !important; }

    /* ── Hide branding ── */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_QUESTIONS = [
    "What is the difference between Dimensions and Measures?",
    "How to create a calculated field in Tableau?",
    "What color represents Measures in the interface?",
    "How do I connect Tableau to a SQL database?",
    "What is a LOD expression and when should I use it?",
]

METRIC_INFO = {
    "faithfulness": {
        "label": "Faithfulness",
        "help": "Does the answer stay true to the retrieved context? (no hallucination)",
        "icon": "🔒",
    },
    "answer_relevancy": {
        "label": "Answer Relevancy",
        "help": "Is the answer relevant to the question asked?",
        "icon": "🎯",
    },
}


def score_color(val: float) -> str:
    if val >= 0.75:
        return "score-good"
    if val >= 0.5:
        return "score-mid"
    return "score-bad"


def score_emoji(val: float) -> str:
    if val >= 0.75:
        return "🟢"
    if val >= 0.5:
        return "🟡"
    return "🔴"


@st.cache_resource(show_spinner=False)
def load_models():
    """Reuse the same models as app.py."""
    if not GROQ_API_KEY:
        st.error("⚠️ GROQ_API_KEY missing from .env")
        st.stop()
    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        max_tokens=1024,
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return llm, embeddings


def collect_results(questions: list[str], retriever, rag_chain) -> list[dict]:
    """Run each question through the RAG pipeline and collect answers + contexts."""
    results = []
    errors = []

    progress = st.progress(0, text="Collecting answers…")
    for i, question in enumerate(questions):
        try:
            # Retrieve context
            docs = retriever.invoke(question)
            context = [doc.page_content for doc in docs]

            # Generate answer (no chat history for eval)
            response = rag_chain.invoke({"input": question, "chat_history": []})
            answer = response.get("answer", "")

            results.append({
                "question": question,
                "answer": answer,
                "contexts": context,
                # ground_truth omitted — unsupervised eval
            })
        except Exception as e:
            errors.append(f"Q{i+1}: {e}")
            results.append({
                "question": question,
                "answer": "ERROR",
                "contexts": [""],
            })

        progress.progress((i + 1) / len(questions), text=f"Processing {i+1}/{len(questions)}…")

    progress.empty()

    if errors:
        st.warning(f"⚠️ {len(errors)} question(s) failed:\n" + "\n".join(errors))

    return results


def run_ragas(results: list[dict], llm, embeddings) -> pd.DataFrame:
    """Run RAGAS evaluation and return a DataFrame."""
    dataset = Dataset.from_list(results)
    score = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )
    return score.to_pandas()


def export_report(df: pd.DataFrame, questions: list[str]) -> dict:
    """Build a JSON-serializable report."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "num_questions": len(questions),
        "averages": {
            col: round(float(df[col].mean()), 4)
            for col in ["faithfulness", "answer_relevancy"]
            if col in df.columns
        },
        "rows": df.to_dict(orient="records"),
    }
    return report


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧪 Eval Config")
    st.markdown("---")

    # System status
    has_retriever = "retriever" in st.session_state and st.session_state.retriever is not None
    has_chain = "rag_chain" in st.session_state and st.session_state.rag_chain is not None

    if has_retriever and has_chain:
        st.markdown('<span style="color:#15803d;font-weight:600">● System ready (from app.py)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#b45309;font-weight:600">● No RAG system detected</span>', unsafe_allow_html=True)
        st.caption("Open **app.py** first and build the knowledge base, then come back here.")

    st.markdown("---")
    st.markdown("### 📝 Test Questions")

    # Question editor
    raw_text = st.text_area(
        "One question per line",
        value="\n".join(DEFAULT_QUESTIONS),
        height=200,
        help="Edit, add, or remove questions for the evaluation.",
    )
    questions = [q.strip() for q in raw_text.strip().split("\n") if q.strip()]
    st.caption(f"{len(questions)} question(s) loaded")

    st.markdown("---")
    st.markdown("### ⚙️ Options")
    show_raw = st.toggle("Show raw answers", value=False)
    show_contexts = st.toggle("Show retrieved contexts", value=False)
    auto_export = st.toggle("Auto-save report as JSON", value=True)

    st.markdown("---")
    run_btn = st.button(
        "🚀 Run Evaluation",
        disabled=not (has_retriever and has_chain and len(questions) > 0),
        use_container_width=True,
        type="primary",
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🧪 RAG Evaluation Dashboard")
st.markdown("Evaluate your Tableau Expert RAG pipeline using **RAGAS** — no ground truth needed.")
st.markdown("---")

# ── Empty state ───────────────────────────────────────────────────────────────
if not has_retriever or not has_chain:
    st.info(
        "**To use this page:**\n\n"
        "1. Run `streamlit run app.py`\n"
        "2. Upload your Tableau PDF and click **Build Knowledge Base**\n"
        "3. Come back here and click **Run Evaluation**",
        icon="ℹ️",
    )
    st.markdown("---")
    st.markdown("#### Preview — Questions that will be evaluated:")
    for i, q in enumerate(questions, 1):
        st.markdown(f'<div class="question-chip">**{i}.** {q}</div>', unsafe_allow_html=True)
    st.stop()

# ── Evaluation run ────────────────────────────────────────────────────────────
if run_btn:
    llm, embeddings = load_models()

    with st.status("Running evaluation…", expanded=True) as status:

        # Step 1: collect
        st.write("📥 Step 1 — Collecting answers from RAG pipeline…")
        t0 = time.time()
        results = collect_results(
            questions,
            st.session_state.retriever,
            st.session_state.rag_chain,
        )
        t1 = time.time()
        st.write(f"✅ Answers collected in {t1-t0:.1f}s")

        # Step 2: RAGAS
        st.write("🔬 Step 2 — Computing RAGAS scores (this takes ~1 min)…")
        try:
            df = run_ragas(results, llm, embeddings)
            t2 = time.time()
            st.write(f"✅ Scores computed in {t2-t1:.1f}s")
            st.session_state["eval_df"] = df
            st.session_state["eval_results"] = results
            st.session_state["eval_questions"] = questions
        except Exception as e:
            status.update(label="❌ Evaluation failed", state="error")
            st.error(f"RAGAS error: {e}")
            st.code(traceback.format_exc())
            st.stop()

        # Step 3: export
        if auto_export:
            report = export_report(df, questions)
            report_path = Path("eval_report.json")
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            st.write(f"💾 Report saved → `{report_path.resolve()}`")

        status.update(label="✅ Evaluation complete!", state="complete")

# ── Results display ───────────────────────────────────────────────────────────
if "eval_df" in st.session_state:
    df = st.session_state["eval_df"]
    results = st.session_state.get("eval_results", [])

    # ── Metric summary cards ──────────────────────────────────────────────────
    st.markdown("### 📊 Overall Scores")
    metric_cols = [c for c in ["faithfulness", "answer_relevancy"] if c in df.columns]
    cols = st.columns(len(metric_cols))

    for col, metric in zip(cols, metric_cols):
        info = METRIC_INFO.get(metric, {"label": metric, "help": "", "icon": "📌"})
        avg = df[metric].mean()
        css_cls = score_color(avg)
        with col:
            st.markdown(
                f'<div class="score-card">'
                f'<div style="font-size:1.8em">{info["icon"]}</div>'
                f'<div style="color:#6b7280;font-size:0.85em;margin-top:4px">{info["label"]}</div>'
                f'<div class="score-value {css_cls}">{avg:.2f}</div>'
                f'<div style="color:#9ca3af;font-size:0.75em">{score_emoji(avg)} {"Good" if avg>=0.75 else "Fair" if avg>=0.5 else "Poor"}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.caption(info["help"])

    st.markdown("---")

    # ── Per-question table ────────────────────────────────────────────────────
    st.markdown("### 📋 Per-Question Breakdown")

    display_cols = ["user_input"] + metric_cols
    styled_df = df[display_cols].copy()
    for m in metric_cols:
        styled_df[m] = styled_df[m].apply(lambda x: f"{score_emoji(x)} {x:.3f}")

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("### 📈 Score Distribution")
    chart_df = df[metric_cols].copy()
    chart_df.index = [f"Q{i+1}" for i in range(len(chart_df))]
    st.bar_chart(chart_df)

    # ── Raw answers & contexts ─────────────────────────────────────────────────
    if show_raw or show_contexts:
        st.markdown("### 🔍 Detailed Results")
        for i, row in enumerate(results):
            with st.expander(f"Q{i+1}: {row['question'][:80]}…"):
                if show_raw:
                    st.markdown("**Answer:**")
                    st.write(row.get("answer", "—"))
                if show_contexts:
                    st.markdown("**Retrieved contexts:**")
                    for j, ctx in enumerate(row.get("contexts", []), 1):
                        st.markdown(
                            f'<div style="background:#f0f4ff;border-left:3px solid #4361ee;'
                            f'border-radius:6px;padding:8px 12px;margin:4px 0;font-size:0.82em;color:#475569">'
                            f'<strong>Chunk #{j}</strong><br>{ctx[:400]}{"…" if len(ctx)>400 else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # ── Export buttons ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💾 Export")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.download_button(
            "📥 Download CSV",
            data=df.to_csv(index=False),
            file_name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_b:
        report = export_report(df, st.session_state.get("eval_questions", []))
        st.download_button(
            "📥 Download JSON",
            data=json.dumps(report, ensure_ascii=False, indent=2),
            file_name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_c:
        st.download_button(
            "📥 Download Markdown",
            data=df[display_cols].to_markdown(index=False),
            file_name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )