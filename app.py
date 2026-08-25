import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.sammarize import summarize, generate_title
from core.transcriber import transcribe_all
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    :root {
        --accent: #6C5CE7;
        --accent-soft: #A29BFE;
        --bg-card: #ffffff;
        --border-soft: #ECECF7;
    }

    /* Overall page */
    .stApp {
        background: linear-gradient(180deg, #FAFAFF 0%, #F3F2FB 100%);
    }

    /* Hide default hamburger footer clutter a bit */
    footer {visibility: hidden;}

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #14121F;
    }
    section[data-testid="stSidebar"] * {
        color: #EDEBFA !important;
    }
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background-color: #221F33 !important;
        border: 1px solid #362F52 !important;
        border-radius: 10px !important;
        color: #EDEBFA !important;
    }
    section[data-testid="stSidebar"] .stRadio > label,
    section[data-testid="stSidebar"] label {
        color: #C9C4E8 !important;
    }
    section[data-testid="stSidebar"] .stFileUploader {
        background-color: #221F33;
        border: 1px dashed #4A4270;
        border-radius: 12px;
        padding: 0.5rem;
    }

    /* Sidebar title block */
    .sidebar-title {
        font-size: 1.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #A29BFE, #74B9FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sidebar-caption {
        color: #9089B8 !important;
        font-size: 0.85rem;
        margin-bottom: 1.2rem;
    }

    /* Primary button */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #6C5CE7, #A29BFE);
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.65rem 1rem;
        box-shadow: 0 6px 16px rgba(108, 92, 231, 0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(108, 92, 231, 0.45);
    }
    div.stButton > button:not([kind="primary"]) {
        border-radius: 10px;
        border: 1px solid #362F52;
        background: #221F33;
        color: #EDEBFA;
    }

    /* Hero header */
    .hero {
        background: linear-gradient(120deg, #6C5CE7 0%, #A29BFE 55%, #74B9FF 100%);
        border-radius: 20px;
        padding: 2rem 2.2rem;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 12px 30px rgba(108, 92, 231, 0.25);
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        line-height: 1.25;
    }
    .hero-sub {
        font-size: 0.95rem;
        opacity: 0.9;
        margin: 0;
    }

    /* Empty state card */
    .empty-state {
        background: var(--bg-card);
        border: 1px solid var(--border-soft);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(108, 92, 231, 0.08);
    }
    .empty-state h2 {
        margin-bottom: 0.4rem;
    }
    .empty-state p {
        color: #6B6785;
    }

    /* Metric-style stat chips */
    .stat-row {
        display: flex;
        gap: 0.8rem;
        margin-bottom: 1.4rem;
        flex-wrap: wrap;
    }
    .stat-chip {
        background: var(--bg-card);
        border: 1px solid var(--border-soft);
        border-radius: 14px;
        padding: 0.7rem 1.1rem;
        box-shadow: 0 4px 14px rgba(108, 92, 231, 0.06);
        flex: 1;
        min-width: 140px;
    }
    .stat-chip .label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #8A85AD;
        font-weight: 700;
    }
    .stat-chip .value {
        font-size: 1.05rem;
        font-weight: 700;
        color: #2D2A45;
    }

    /* Content card wrapping each tab's body */
    .content-card {
        background: var(--bg-card);
        border: 1px solid var(--border-soft);
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 8px 24px rgba(108, 92, 231, 0.06);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #EFEDFB;
        padding: 6px;
        border-radius: 14px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 16px;
        font-weight: 600;
        color: #6B6785;
    }
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #6C5CE7 !important;
        box-shadow: 0 4px 10px rgba(108, 92, 231, 0.15);
    }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 0.3rem 0.2rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing" not in st.session_state:
    st.session_state.processing = False


def run_pipeline(source: str, language: str) -> dict:
    """Same logic as the CLI version, wrapped for Streamlit."""
    chunks = process_input(source)
    transcript = transcribe_all(chunks, language=language)
    title = generate_title(transcript)
    summary = summarize(transcript)
    action_items = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">🎬 AI Video Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-caption">Turn a video/audio source into a transcript, '
        'summary, and a chatbot.</div>',
        unsafe_allow_html=True,
    )

    input_mode = st.radio("Source type", ["YouTube URL", "Upload local file"])

    source = None
    uploaded_file = None

    if input_mode == "YouTube URL":
        source = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
    else:
        uploaded_file = st.file_uploader(
            "Upload audio/video file", type=["mp4", "mp3", "wav", "m4a", "mov", "mkv"]
        )
        if uploaded_file is not None:
            import tempfile, os
            suffix = os.path.splitext(uploaded_file.name)[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded_file.read())
            tmp.close()
            source = tmp.name

    language = st.selectbox("Language", ["english", "hinglish"])

    run_clicked = st.button(
        "🚀 Run Pipeline", type="primary", use_container_width=True,
        disabled=st.session_state.processing,
    )

    if st.session_state.result:
        st.divider()
        if st.button("🗑️ Clear session", use_container_width=True):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.rerun()

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
if run_clicked:
    if not source:
        st.sidebar.error("Please provide a YouTube URL or upload a file.")
    else:
        st.session_state.processing = True
        st.session_state.chat_history = []
        with st.spinner("Processing source, transcribing, and analyzing... this can take a while ⏳"):
            try:
                st.session_state.result = run_pipeline(source, language)
                st.toast("Done! ✅")
            except Exception as e:
                st.session_state.result = None
                st.error(f"Pipeline failed: {e}")
        st.session_state.processing = False

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
result = st.session_state.result

if result is None:
    st.markdown(
        """
        <div class="hero">
            <p class="hero-title">🎬 AI Video Assistant</p>
            <p class="hero-sub">Drop in a YouTube link or a local recording — get a transcript,
            summary, action items, key decisions, and a chatbot that knows the whole video.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="empty-state">
            <h2>👋 Ready when you are</h2>
            <p>Add a source in the sidebar and click <b>Run Pipeline</b> to get started.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    word_count = len(result["transcript"].split())
    read_minutes = max(1, word_count // 130)

    st.markdown(
        f"""
        <div class="hero">
            <p class="hero-title">📌 {result['title']}</p>
            <p class="hero-sub">Language: {language.capitalize()} · Generated from your source</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="stat-row">
            <div class="stat-chip"><div class="label">Transcript length</div>
                <div class="value">{word_count:,} words</div></div>
            <div class="stat-chip"><div class="label">Est. duration</div>
                <div class="value">~{read_minutes} min read</div></div>
            <div class="stat-chip"><div class="label">Language</div>
                <div class="value">{language.capitalize()}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_summary, tab_transcript, tab_actions, tab_decisions, tab_questions, tab_chat = st.tabs(
        ["📋 Summary", "📝 Transcript", "✅ Action Items", "🔑 Key Decisions", "❓ Open Questions", "💬 Chat"]
    )

    with tab_summary:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown(result["summary"])
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_transcript:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.text_area("Full transcript", result["transcript"], height=460, label_visibility="collapsed")
        st.download_button(
            "⬇️ Download transcript (.txt)",
            result["transcript"],
            file_name="transcript.txt",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_actions:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown(result["action_items"])
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_decisions:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown(result["key_decisions"])
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_questions:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown(result["open_questions"])
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_chat:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.caption("💡 Ask questions about the video — answered via RAG over the transcript.")

        chat_container = st.container(height=420)
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown(
                    "<p style='color:#9089B8;'>No messages yet — ask something like "
                    "<i>'What were the main takeaways?'</i></p>",
                    unsafe_allow_html=True,
                )
            for role, msg in st.session_state.chat_history:
                with st.chat_message(role):
                    st.markdown(msg)

        question = st.chat_input("Ask something about this video...")
        if question:
            st.session_state.chat_history.append(("user", question))
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            answer = ask_question(result["rag_chain"], question)
                        except Exception as e:
                            answer = f"⚠️ Error answering question: {e}"
                        st.markdown(answer)
            st.session_state.chat_history.append(("assistant", answer))
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)