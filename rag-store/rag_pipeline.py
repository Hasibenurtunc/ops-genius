from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

def load_documents():
    docs = []
    docs_path = "./docs"
    for filename in os.listdir(docs_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(docs_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                docs.append(Document(
                    page_content=content,
                    metadata={"source": filename}
                ))
            print(f"✅ Yüklendi: {filename}")
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"\n📄 Toplam chunk sayısı: {len(chunks)}")
    return chunks

def create_vectorstore(chunks):
    print("\n🔄 Embedding oluşturuluyor...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ ChromaDB oluşturuldu!")
    return vectorstore

def create_rag_pipeline(vectorstore):
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

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

def main():
    print("🚀 RAG Pipeline başlatılıyor...\n")

    docs = load_documents()
    chunks = chunk_documents(docs)
    vectorstore = create_vectorstore(chunks)
    rag_chain, retriever = create_rag_pipeline(vectorstore)

    print("\n✅ Sistem hazır! Test soruları çalıştırılıyor...\n")
    print("=" * 60)

    test_questions = [
        "How can I cancel my order?",
        "What is the refund policy?",
        "How do I recover my password?"
    ]

    for question in test_questions:
        print(f"\n❓ Soru: {question}")
        answer = rag_chain.invoke(question)
        sources = retriever.invoke(question)
        print(f"💬 Cevap: {answer}")
        print(f"📄 Kaynak: {sources[0].metadata['source']}")
        print("-" * 60)

if __name__ == "__main__":
    main()