from dotenv import load_dotenv
import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ============================================================
# RAG SİSTEMİ YÜKLE
# ============================================================
def load_rag():
    print(" RAG sistemi yükleniyor...")
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
    print(" Sistem hazır!\n")
    return chain, retriever

# ============================================================
# KONTROLLÜ TEST — VERİ SETİNDEKİ SORULAR AYNEN
# ============================================================
def run_controlled_test():
    df = pd.read_csv("phase2/filtered_dataset.csv")
    chain, retriever = load_rag()

    results = []
    print("=" * 60)
    print("🧪 AŞAMA 1: KONTROLLÜ TEST")
    print("Veri setindeki sorular aynen sisteme soruluyor...")
    print("=" * 60)

    for i, row in df.iterrows():
        question   = row["instruction"]
        ground_truth = row["response"]
        intent     = row["intent"]
        flags      = row["flags"]

        # Placeholder temizle
        clean_q = question.replace("{{Order Number}}", "12345") \
                  .replace("{{Account Type}}", "standard") \
                  .replace("{{Account Category}}", "premium") \
                  .replace("{{Name}}", "John") \
                  .replace("{{Delivery City}}", "London") \
                  .replace("{{Delivery Country}}", "United Kingdom") \
                  .replace("{{Order Number}}", "12345") \
                  .replace("purchase 12345", "my order") \
                  .replace("order 12345", "my order")

        # RAG cevabı al
        rag_answer = chain.invoke(clean_q)
        sources = retriever.invoke(clean_q)
        source_doc = sources[0].metadata["source"] if sources else "N/A"

        # Başarı kontrolü
        failed = "could not find" in rag_answer.lower()
        status = " FAIL" if failed else " OK"

        print(f"\n[{i+1}] {status} | Intent: {intent} | Flags: {flags}")
        print(f"Soru   : {clean_q[:80]}...")
        print(f"Cevap  : {rag_answer[:120]}...")
        print(f"Kaynak : {source_doc}")
        print("-" * 60)

        results.append({
            "test_type": "controlled",
            "intent": intent,
            "flags": flags,
            "original_question": question,
            "clean_question": clean_q,
            "rag_answer": rag_answer,
            "ground_truth": ground_truth,
            "source": source_doc,
            "failed": failed
        })

    # Sonuçları kaydet
    results_df = pd.DataFrame(results)
    results_df.to_csv("phase2/results/controlled_results.csv", index=False)

    # Özet
    total = len(results_df)
    success = len(results_df[results_df["failed"] == False])
    fail = len(results_df[results_df["failed"] == True])

    print("\n" + "=" * 60)
    print("KONTROLLÜ TEST SONUÇLARI")
    print("=" * 60)
    print(f"Toplam soru : {total}")
    print(f" Başarılı : {success} (%{round(success/total*100, 1)})")
    print(f" Başarısız: {fail} (%{round(fail/total*100, 1)})")
    print("\nIntent bazında başarı:")
    for intent in results_df["intent"].unique():
        sub = results_df[results_df["intent"] == intent]
        s = len(sub[sub["failed"] == False])
        print(f"  {intent}: {s}/{len(sub)} başarılı")
    print(f"\n Kaydedildi: phase2/results/controlled_results.csv")

if __name__ == "__main__":
    run_controlled_test()