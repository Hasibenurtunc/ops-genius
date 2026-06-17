from dotenv import load_dotenv
import os
import pandas as pd
from datasets import load_dataset
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ============================================================
# 1. VECTORSTORE YÜKLE (tekrar oluşturma!)
# ============================================================
def load_vectorstore():
    print("🔄 ChromaDB yükleniyor...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    print("✅ ChromaDB yüklendi!")
    return vectorstore

# ============================================================
# 2. RAG PIPELINE
# ============================================================
def create_rag_pipeline(vectorstore):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = PromptTemplate.from_template("""
You are a helpful customer support assistant.
Answer the question based ONLY on the following context.
If the answer is not in the context, say "I could not find information about this in our policy documents."

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

# ============================================================
# 3. BİTEXT'TEN SORULARI AL
# ============================================================
def load_bitext_samples(n_per_intent=2):
    print("\n📦 Bitext yükleniyor...")
    dataset = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    )
    df = dataset["train"].to_pandas()

    # Her intent'ten 2 örnek al
    samples = df.groupby("intent").apply(
        lambda x: x.head(n_per_intent)
    ).reset_index(drop=True)

    print(f"✅ {len(samples)} soru seçildi ({df['intent'].nunique()} intent)")
    return samples

# ============================================================
# 4. TEST ET VE KAYDET
# ============================================================
def run_tests(rag_chain, retriever, samples):
    results = []

    print("\n🧪 Test başlıyor...\n")
    print("=" * 70)

    for i, row in samples.iterrows():
        question = row["instruction"]
        ground_truth = row["response"]
        intent = row["intent"]

        # RAG cevabı al
        rag_answer = rag_chain.invoke(question)
        sources = retriever.invoke(question)
        source_doc = sources[0].metadata["source"] if sources else "N/A"

        print(f"[{i+1}] Intent: {intent}")
        print(f"❓ Soru     : {question[:80]}...")
        print(f"💬 RAG Cevap: {rag_answer[:150]}...")
        print(f"✅ GT Cevap : {ground_truth[:150]}...")
        print(f"📄 Kaynak   : {source_doc}")
        print("-" * 70)

        results.append({
            "intent": intent,
            "question": question,
            "rag_answer": rag_answer,
            "ground_truth": ground_truth,
            "source": source_doc
        })

    # Sonuçları CSV'ye kaydet
    results_df = pd.DataFrame(results)
    results_df.to_csv("test_results.csv", index=False)
    print(f"\n💾 Sonuçlar 'test_results.csv' dosyasına kaydedildi!")
    return results_df

# ============================================================
# 5. MAIN
# ============================================================
def main():
    vectorstore = load_vectorstore()
    rag_chain, retriever = create_rag_pipeline(vectorstore)
    samples = load_bitext_samples(n_per_intent=2)
    results = run_tests(rag_chain, retriever, samples)
    print(f"\n✅ Test tamamlandı! Toplam {len(results)} soru test edildi.")

if __name__ == "__main__":
    main()
    
    
    
    
# rag_pipeline.py    → Sistemi kur, veritabanını oluştur
# test_with_bitext.py → Sistemi test et, sonuçları kaydet    