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
.stApp { background-color: #f0f4f8; }
[data-testid="stSidebar"] { background-color: #1a2b4a !important; }
[data-testid="stSidebar"] * { color: white !important; }
.sidebar-header { padding: 24px 20px 16px; border-bottom: 1px solid rgba(255,255,255,0.1); }
.sidebar-logo { font-size: 22px; font-weight: 700; color: white; margin-bottom: 4px; }
.sidebar-tagline { font-size: 11px; color: rgba(255,255,255,0.5); letter-spacing: 1.5px; text-transform: uppercase; }
.bot-message {
    background: #f8fafc;
    border: 1px solid #e8edf2;
    border-radius: 0 16px 16px 16px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 75%;
    font-size: 14px;
    line-height: 1.6;
    color: #1e293b;
}
.user-message {
    background: #1a2b4a;
    border-radius: 16px 16px 0 16px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 75%;
    font-size: 14px;
    line-height: 1.6;
    color: white;
    margin-left: auto;
    text-align: right;
}
.source-tag {
    font-size: 11px;
    color: #94a3b8;
    margin-top: 4px;
    margin-bottom: 12px;
}
.chat-header {
    background: white;
    border-radius: 16px;
    padding: 16px 24px;
    margin-bottom: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.powered-by {
    text-align: center;
    font-size: 10px;
    color: #94a3b8;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 12px;
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
    st.markdown('<div style="padding:0 8px;font-size:11px;color:rgba(255,255,255,0.4);letter-spacing:1px;text-transform:uppercase;">Quick Actions</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    quick_actions = [
        ("📦", "Cancel Order", "How can I cancel my order?"),
        ("↩️", "Return & Refund", "What is the refund policy?"),
        ("👤", "Account Help", "How do I recover my password?"),
        ("🚚", "Delivery Info", "What delivery options are available?"),
    ]

    for icon, label, question in quick_actions:
        if st.button(f"{icon}  {label}", key=f"btn_{label}", use_container_width=True):
            st.session_state.pending_question = question
            st.rerun()

# Ana alan
st.markdown("""
<div class="chat-header">
    <div style="font-weight:600;font-size:16px;color:#1e293b;">Ops-Genius Assistant</div>
    <div style="font-size:12px;color:#22c55e;">● Online</div>
</div>
""", unsafe_allow_html=True)

# İlk mesaj
if not st.session_state.messages:
    st.markdown("""
    <div class="bot-message">
        Hello! How can I help you today? 👋<br><br>
        You can ask me about your orders, refunds, account issues, or delivery options.
    </div>
    <div class="source-tag">🤖 Ops-Genius Assistant</div>
    """, unsafe_allow_html=True)

# Mesaj geçmişi
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("source"):
            st.markdown(f'<div class="source-tag">📄 Source: {msg["source"]}</div>', unsafe_allow_html=True)

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