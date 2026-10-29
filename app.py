import os
import streamlit as st

from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

st.set_page_config(
    page_title="Campus AI | College Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #0b1020;
        --panel: #111a30;
        --panel-light: #17223d;
        --border: rgba(148, 163, 184, 0.17);
        --text: #f1f5f9;
        --muted: #94a3b8;
        --purple: #8b5cf6;
        --blue: #60a5fa;
        --green: #34d399;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(99,102,241,.14), transparent 38%),
            radial-gradient(ellipse at 90% 20%, rgba(139,92,246,.12), transparent 35%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 16, 32, .75);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10182c, #0b1020);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 0 0 28px 0;
    }

    .brand-icon {
        width: 48px;
        height: 48px;
        display: flex;
        justify-content: center;
        align-items: center;
        border-radius: 15px;
        font-size: 25px;
        background: linear-gradient(135deg, #7c3aed, #3b82f6);
        box-shadow: 0 8px 28px rgba(124,58,237,.28);
    }

    .brand-name {
        font-size: 20px;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -.5px;
    }

    .brand-subtitle {
        color: #94a3b8;
        font-size: 11px;
        margin-top: 3px;
    }

    .sidebar-label {
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 10px;
        font-weight: 700;
        margin: 20px 0 10px;
    }

    .sidebar-card {
        padding: 15px;
        border: 1px solid var(--border);
        border-radius: 15px;
        background: rgba(255,255,255,.035);
        margin: 10px 0;
    }

    .sidebar-card-title {
        color: #e2e8f0;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 5px;
    }

    .sidebar-card-desc {
        color: #94a3b8;
        font-size: 11px;
        line-height: 1.6;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 35px 32px;
        border-radius: 25px;
        border: 1px solid rgba(167,139,250,.23);
        background:
            radial-gradient(circle at 85% 10%, rgba(96,165,250,.22), transparent 32%),
            linear-gradient(120deg, #191b3b 0%, #172746 52%, #17223d 100%);
        margin-bottom: 22px;
    }

    .hero-eyebrow {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 30px;
        background: rgba(167,139,250,.13);
        border: 1px solid rgba(167,139,250,.28);
        color: #c4b5fd;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 17px;
    }

    .hero h1 {
        color: #ffffff;
        font-size: clamp(27px, 3.2vw, 42px);
        font-weight: 800;
        letter-spacing: -1.5px;
        line-height: 1.2;
        margin: 0 0 12px;
    }

    .hero h1 span {
        background: linear-gradient(90deg, #c4b5fd, #93c5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        color: #bdc9df;
        font-size: 14px;
        line-height: 1.8;
        max-width: 610px;
        margin: 0;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 7px 12px;
        border: 1px solid rgba(52,211,153,.25);
        border-radius: 30px;
        background: rgba(52,211,153,.08);
        color: #6ee7b7;
        font-size: 11px;
        font-weight: 600;
        margin-top: 20px;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        background: #34d399;
        border-radius: 50%;
        box-shadow: 0 0 10px rgba(52,211,153,.65);
    }

    .section-heading {
        font-size: 17px;
        font-weight: 700;
        color: #f1f5f9;
        margin: 18px 0 4px;
    }

    .section-subheading {
        font-size: 12px;
        color: #94a3b8;
        margin-bottom: 15px;
    }

    .feature-card {
        background: linear-gradient(145deg, rgba(30,41,70,.88), rgba(17,26,48,.88));
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 20px 17px;
        min-height: 145px;
        transition: transform .2s, border-color .2s;
    }

    .feature-card:hover {
        transform: translateY(-3px);
        border-color: rgba(139,92,246,.5);
    }

    .feature-icon {
        font-size: 25px;
        margin-bottom: 13px;
    }

    .feature-title {
        color: #f1f5f9;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 7px;
    }

    .feature-desc {
        color: #94a3b8;
        font-size: 11px;
        line-height: 1.7;
    }

    .chat-heading {
        font-size: 19px;
        font-weight: 750;
        color: #f8fafc;
        margin-top: 30px;
        margin-bottom: 5px;
    }

    .chat-subtitle {
        color: #94a3b8;
        font-size: 12px;
        margin-bottom: 18px;
    }

    [data-testid="stChatMessage"] {
        background: rgba(23,34,61,.66);
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 15px 18px;
        margin-bottom: 12px;
    }

    [data-testid="stChatMessage"]:hover {
        border-color: rgba(139,92,246,.24);
    }

    [data-testid="stChatInput"] {
        border: 1px solid rgba(139,92,246,.38);
        border-radius: 17px;
        background: #111a30;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: #8b5cf6;
        box-shadow: 0 0 0 2px rgba(139,92,246,.12);
    }

    [data-testid="stChatInput"] textarea {
        color: #f1f5f9 !important;
    }

    .query-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 30px;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: .6px;
        margin-bottom: 8px;
    }

    .badge-academic {
        background: rgba(96,165,250,.12);
        border: 1px solid rgba(96,165,250,.25);
        color: #93c5fd;
    }

    .badge-fee {
        background: rgba(251,191,36,.10);
        border: 1px solid rgba(251,191,36,.25);
        color: #fcd34d;
    }

    .badge-general {
        background: rgba(52,211,153,.10);
        border: 1px solid rgba(52,211,153,.25);
        color: #6ee7b7;
    }

    div.stButton > button {
        border: 1px solid var(--border);
        border-radius: 12px;
        background: rgba(255,255,255,.035);
        color: #e2e8f0;
        font-weight: 600;
        transition: all .2s ease;
        min-height: 42px;
    }

    div.stButton > button:hover {
        border-color: #8b5cf6;
        background: rgba(139,92,246,.12);
        color: #ffffff;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #7c3aed, #4f46e5);
        color: #ffffff;
        border: none;
    }

    hr {
        border-color: var(--border);
    }

    .footer {
        color: #64748b;
        font-size: 11px;
        text-align: center;
        padding: 25px 0 12px;
    }

    @media (max-width: 700px) {
        .hero {
            padding: 25px 20px;
        }

        .hero h1 {
            letter-spacing: -.7px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Preparing your college knowledge base...")
def load_resources():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    def build_retriever(pdf_path: str):
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(
                f"Could not find {pdf_path}. "
                "Place the PDF in the project folder or update its path."
            )

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if not documents:
            raise ValueError(f"No pages could be read from {pdf_path}.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError(f"No text chunks extracted from {pdf_path}.")

        vectorstore = FAISS.from_documents(chunks, embeddings)

        return vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

    # Update these paths to match your actual filenames.
    academic_retriever = build_retriever("academics_handbook.pdf")
    fee_retriever = build_retriever("fee_structure.pdf")

    # Use a model ID that is currently available to your Groq account.
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.4,
        max_retries=2,
    )

    return academic_retriever, fee_retriever, llm


try:
    academic_retriever, fee_retriever, llm = load_resources()
except Exception as exc:
    st.error(f"Unable to initialize the assistant: {exc}")
    st.info(
        "Check your PDF filenames, installed dependencies, "
        "Groq API key, and available model ID."
    )
    st.stop()

class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


def classifier_node(state: State) -> dict:
    query = state["messages"][-1].content.strip()
    normalized = query.lower().strip("!?., ")

    greetings = {
        "hi",
        "hello",
        "hey",
        "how are you",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
    }

    # Skip the classification API call for common greetings.
    if normalized in greetings:
        return {"query_type": "general"}

    prompt = f"""
Classify the student query into exactly one category:
academic, fee, or general.

academic: attendance, examinations, grading, credits, promotion,
course structure, summer training, degree requirements.

fee: tuition, payment, refunds, late charges, scholarships,
or other money-related college matters.

general: greetings, casual conversation, or unrelated questions.

Query: {query}

Return exactly one word: academic, fee, or general.
"""

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"

    return {"query_type": category}


def academic_rag_node(state: State) -> dict:
    query = state["messages"][-1].content
    docs = academic_retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    return {"retrieved_context": context}


def fee_rag_node(state: State) -> dict:
    query = state["messages"][-1].content
    docs = fee_retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    return {"retrieved_context": context}


def general_node(state: State) -> dict:
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def response_node(state: State) -> dict:
    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = f"""
You are a friendly college assistant speaking with a
{programme} student.

Respond naturally and helpfully to the following message.
Do not invent college-specific policies or fee details.

Student message: {query}
"""
    else:
        prompt = f"""
You are a college assistant helping a {programme} student.

Answer using ONLY the college document context below.
If the context does not contain sufficient information,
clearly say that the provided documents do not specify the answer.
Never invent fees, deadlines, rules, or policies.

Prioritize programme-specific information relevant to {programme}.

DOCUMENT CONTEXT:
{context}

STUDENT QUESTION:
{query}

Give a clear, friendly, concise answer.
"""

    response = llm.invoke(prompt)

    return {
        "messages": [
            ("ai", response.content.strip())
        ]
    }


def route_query(state: State) -> str:
    query_type = state["query_type"]

    if query_type == "academic":
        return "academic_rag"

    if query_type == "fee":
        return "fee_rag"

    return "general"


@st.cache_resource(show_spinner=False)
def build_graph():
    graph = StateGraph(State)

    graph.add_node("classifier", classifier_node)
    graph.add_node("academic_rag", academic_rag_node)
    graph.add_node("fee_rag", fee_rag_node)
    graph.add_node("general", general_node)
    graph.add_node("response", response_node)

    graph.add_edge(START, "classifier")

    graph.add_conditional_edges(
        "classifier",
        route_query,
    )

    graph.add_edge("academic_rag", "response")
    graph.add_edge("fee_rag", "response")
    graph.add_edge("general", "response")

    graph.add_edge("response", END)

    return graph.compile()


app = build_graph()

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🎓</div>
            <div>
                <div class="brand-name">Campus AI</div>
                <div class="brand-subtitle">YOUR SMART COLLEGE COMPANION</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-label">Personalization</div>',
        unsafe_allow_html=True,
    )

    student_programme = st.selectbox(
        "Your programme",
        options=["BCA", "BBA", "B.Com (H)"],
    )

    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-card-title">👤 Student Profile</div>
            <div class="sidebar-card-desc">
                Your assistant is personalized for
                <b>{student_programme}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-label">Knowledge Sources</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-card-title">📘 Academic Handbook</div>
            <div class="sidebar-card-desc">
                Attendance, exams, credits and academic rules.
            </div>
        </div>

        <div class="sidebar-card">
            <div class="sidebar-card-title">💰 Fee Structure</div>
            <div class="sidebar-card-desc">
                Tuition, payment information and fee policies.
            </div>
        </div>

        <div class="sidebar-card">
            <div class="sidebar-card-title">💬 General Assistant</div>
            <div class="sidebar-card-desc">
                Greetings and general questions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.lc_messages = []
        st.session_state.last_query_type = None
        st.rerun()

    st.markdown("---")

    st.caption("⚡ Powered by LangGraph + Groq")
    st.caption("🔎 Retrieval powered by FAISS")

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">✦ YOUR AI CAMPUS COMPANION</div>

        <h1>Knowledge at your fingertips.<br>
        <span>Answers without the wait.</span></h1>

        <p>
            Your intelligent college assistant for academic regulations,
            fee information and everyday questions.
            Ask naturally, get clear answers.
        </p>

        <div class="status-pill">
            <span class="status-dot"></span>
            AI ASSISTANT READY
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-heading">Everything you need, in one place</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-subheading">'
    'Select a knowledge area or start a conversation below.'
    '</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">📘</div>
            <div class="feature-title">Academic Support</div>
            <div class="feature-desc">
                Understand attendance, examinations,
                grading and degree requirements.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">💳</div>
            <div class="feature-title">Fee Information</div>
            <div class="feature-desc">
                Find fee details and payment rules
                from your college documents.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">✨</div>
            <div class="feature-title">Smart Assistance</div>
            <div class="feature-desc">
                Ask questions naturally and get
                clear, helpful responses.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "lc_messages" not in st.session_state:
    st.session_state.lc_messages = []

if "last_query_type" not in st.session_state:
    st.session_state.last_query_type = None

st.markdown(
    '<div class="chat-heading">💬 Let’s talk</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="chat-subtitle">'
    'Your college questions, answered with context.'
    '</div>',
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown("**✨ Try asking**")

    quick_questions = [
        "What are the attendance requirements?",
        "Explain the fee structure.",
        "What are the examination rules?",
    ]

    prompt_cols = st.columns(3)

    for index, question in enumerate(quick_questions):
        with prompt_cols[index]:
            if st.button(
                question,
                key=f"quick_{index}",
                use_container_width=True,
            ):
                st.session_state.pending_query = question
                st.rerun()

for msg in st.session_state.messages:
    avatar = "🧑‍🎓" if msg["role"] == "user" else "🎓"

    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant" and msg.get("query_type"):
            query_type = msg["query_type"]

            st.markdown(
                f'<span class="query-badge badge-{query_type}">'
                f'{query_type.upper()}'
                f'</span>',
                unsafe_allow_html=True,
            )

        st.markdown(msg["content"])

user_query = st.chat_input(
    "Ask about academics, fees, attendance..."
)

if "pending_query" in st.session_state:
    user_query = st.session_state.pop("pending_query")

if user_query:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_query)

    current_messages = (
        st.session_state.lc_messages
        + [("human", user_query)]
    )

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("✨ Finding the best answer..."):
            try:
                result = app.invoke(
                    {
                        "programme": student_programme,
                        "messages": current_messages,
                    }
                )

                ai_response = result["messages"][-1].content
                query_type = result.get("query_type", "general")

                st.markdown(
                    f'<span class="query-badge badge-{query_type}">'
                    f'{query_type.upper()}'
                    f'</span>',
                    unsafe_allow_html=True,
                )

                st.markdown(ai_response)

                st.session_state.lc_messages = result["messages"]
                st.session_state.last_query_type = query_type

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": ai_response,
                        "query_type": query_type,
                    }
                )

            except Exception as exc:
                st.error(
                    "Sorry, I couldn't process your question. "
                    "Please try again."
                )

                # Keep the details in the terminal rather than
                # exposing internal errors to every user.
                print(f"Assistant error: {exc}")

                # Remove the failed query from the LangGraph
                # conversation history. It can be retried.
                st.session_state.messages.pop()
                
st.markdown(
    """
    <div class="footer">
        CAMPUS AI · Built with Streamlit, LangGraph & FAISS
        <br><br>
        AI-generated answers can contain errors.
        Verify important college policies with official sources.
    </div>
    """,
    unsafe_allow_html=True,
)

