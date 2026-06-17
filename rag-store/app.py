import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(
    page_title="Ops-Genius",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp { background-color: #eef2f7; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a2b4a 0%, #0f1e35 100%) !important;
}
[data-testid="stSidebar"] * { color: white !important; }

.sidebar-header {
    padding: 28px 20px 20px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.sidebar-logo {
    font-size: 22px;
    font-weight: 700;
    color: white;
    margin-bottom: 4px;
}
.sidebar-tagline {
    font-size: 10px;
    color: rgba(255,255,255,0.4);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 6px;
}

.chat-header {
    background: white;
    border-radius: 16px;
    padding: 16px 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.4); }
    70% { box-shadow: 0 0 0 8px rgba(34, 197, 94, 0); }
    100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
}

.pulse-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #22c55e;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}

.message-row-bot {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin: 12px 0;
}

.message-row-user {
    display: flex;
    align-items: flex-start;
    justify-content: flex-end;
    gap: 10px;
    margin: 12px 0;
}

.avatar-bot {
    width: 34px;
    height: 34px;
    background: #e8f0fe;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}

.avatar-user {
    width: 34px;
    height: 34px;
    background: #2E74B5;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}

.bot-message {
    background: white;
    border-left: 3px solid #2E74B5;
    border-radius: 0 16px 16px 16px;
    padding: 12px 16px;
    max-width: 72%;
    font-size: 14px;
    line-height: 1.7;
    color: #1e293b;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

.user-message {
    background: linear-gradient(135deg, #1a2b4a 0%, #2E74B5 100%);
    border-radius: 16px 16px 0 16px;
    padding: 12px 16px;
    max-width: 72%;
    font-size: 14px;
    line-height: 1.7;
    color: white;
    box-shadow: 0 2px 12px rgba(46,116,181,0.3);
}

.source-badge {
    display: inline-block;
    background: #f0f7ff;
    border: 1px solid #bfdbfe;
    color: #3b82f6;
    font-size: 10px;
    padding: 2px 10px;
    border-radius: 20px;
    margin-top: 6px;
    margin-left: 44px;
    letter-spacing: 0.5px;
}

.powered-by {
    text-align: center;
    font-size: 10px;
    color: #94a3b8;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 16px;
    margin-top: 8px;
}

.stButton button {
    border-radius: 10px !important;
    border: none !important;
    background: rgba(255,255,255,0.08) !important;
    color: rgba(255,255,255,0.8) !important;
    text-align: left !important;
    transition: all 0.2s !important;
}

.stButton button:hover {
    background: rgba(255,255,255,0.15) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_rag():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )
    prompt = PromptTemplate.from_template("""
You are a helpful customer support assistant for an e-commerce platform.
Answer the question based ONLY on the following context.
Even if the question uses informal language, slang, abbreviations or
indirect phrasing, try your best to understand the intent and answer
from the context.
If the answer is truly not in the context, say "I could not find
information about this in our policy documents."

Context:
{context}

Question: {question}

Answer:""")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain, retriever

def get_rag_answer(question):
    chain, retriever = load_rag()
    answer = chain.invoke(question)
    sources = retriever.invoke(question)
    source = sources[0].metadata["source"].replace(".txt", "") if sources else "N/A"
    return answer, source

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-logo">🤖 Ops-Genius</div>
        <div class="sidebar-tagline">AI-Powered Customer Support</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("➕  New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="padding:0 8px;font-size:10px;color:rgba(255,255,255,0.35);letter-spacing:2px;text-transform:uppercase;">Quick Actions</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    quick_actions = [
    ("📦", "Cancel Order", "How can I cancel my order?"),
    ("👤", "Account Help", "How do I recover my password?"),
    ("🚚", "Delivery Info", "What delivery options are available?"),
    ]

    for icon, label, question in quick_actions:
        if st.button(f"{icon}  {label}", key=f"btn_{label}", use_container_width=True):
            st.session_state.pending_question = question
            st.rerun()

# Header
st.markdown("""
<div class="chat-header">
    <div style="display:flex;align-items:center;gap:12px;">
        <div style="font-size:28px;">🤖</div>
        <div>
            <div style="font-weight:600;font-size:15px;color:#1e293b;">Ops-Genius Assistant</div>
            <div style="font-size:12px;color:#64748b;margin-top:2px;">
                <span class="pulse-dot"></span>Online
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# İlk karşılama mesajı
if not st.session_state.messages:
    st.markdown("""
    <div class="message-row-bot">
        <div class="avatar-bot">🤖</div>
        <div class="bot-message">
            Hello! How can I help you today? 👋<br><br>
            You can ask me about order cancellations, account & password issues, or delivery options.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Mesaj geçmişi
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="message-row-user">
            <div class="user-message">{msg["content"]}</div>
            <div class="avatar-user">👤</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="message-row-bot">
            <div class="avatar-bot">🤖</div>
            <div class="bot-message">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
        if msg.get("source"):
            st.markdown(f'<div class="source-badge">📄 {msg["source"]}</div>', unsafe_allow_html=True)

# Pending question işle
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
    st.session_state.messages.append({"role": "user", "content": question, "source": None})
    placeholder = st.empty()
    placeholder.info("⏳ Thinking...")
    answer, source = get_rag_answer(question)
    placeholder.empty()
    st.session_state.messages.append({"role": "assistant", "content": answer, "source": source})
    st.rerun()

# Alt bilgi
st.markdown('<div class="powered-by">Powered by RAG Technology • Privacy Policy • Terms of Service</div>', unsafe_allow_html=True)

# Chat input
if question := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": question, "source": None})
    placeholder = st.empty()
    placeholder.info("⏳ Thinking...")
    answer, source = get_rag_answer(question)
    placeholder.empty()
    st.session_state.messages.append({"role": "assistant", "content": answer, "source": source})
    st.rerun()