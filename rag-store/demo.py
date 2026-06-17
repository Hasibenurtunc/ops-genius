from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

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
You are a helpful customer support assistant.
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
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

def ask(question, chain, retriever):
    print(f"\n❓ Soru: {question}")
    answer = chain.invoke(question)
    sources = retriever.invoke(question)
    source = sources[0].metadata["source"] if sources else "N/A"
    print(f"💬 Cevap: {answer}")
    print(f"📄 Kaynak: {source}")
    print("-" * 60)

def main():
    print("🚀 Ops-Genius RAG Sistemi Yükleniyor...")
    chain, retriever = load_rag()
    print("✅ Sistem hazır!\n")
    print("=" * 60)

    # Hazır sorular
    sorular = [
        "How can I cancel my order?",
        "What is the refund policy?",
        "How do I track my order?",
        "What delivery options are available?",
        "How do I recover my password?"
    ]

    for soru in sorular:
        ask(soru, chain, retriever)

    # Canlı soru modu
    print("\n💡 Kendi sorunuzu yazın (çıkmak için 'q'):")
    while True:
        soru = input("\nSoru: ").strip()
        if soru.lower() == 'q':
            break
        if soru:
            ask(soru, chain, retriever)

if __name__ == "__main__":
    main()