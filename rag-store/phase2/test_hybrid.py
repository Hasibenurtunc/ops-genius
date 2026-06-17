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
        | prompt | llm | StrOutputParser()
    )
    return chain, retriever

# Hibrit/Zorlayici sorular
# Birden fazla intent iceriyor ya da cok dolayli ifade kullaniyor
hybrid_questions = [
    {
        "question": "I forgot my password and now I can't track my order or check my delivery options. What should I do first?",
        "intents": ["recover_password", "cancel_order", "delivery_options"],
        "difficulty": "multi-intent",
        "note": "Uc farkli intent ayni anda soruluyor"
    },
    {
        "question": "I placed an order yesterday but I changed my mind. Also I don't remember which email I used to sign up. Can you help?",
        "intents": ["cancel_order", "recover_password"],
        "difficulty": "multi-intent",
        "note": "Iptal + sifre kurtarma ayni anda"
    },
    {
        "question": "My account is locked and I have an urgent delivery coming tomorrow. I need both issues fixed now.",
        "intents": ["recover_password", "delivery_options"],
        "difficulty": "multi-intent + urgency",
        "note": "Acil durum + coklu intent"
    },
    {
        "question": "I no longer wish to proceed with the transaction I initiated earlier today.",
        "intents": ["cancel_order"],
        "difficulty": "very indirect",
        "note": "Cok resmi ve dolayli ifade - cancel order"
    },
    {
        "question": "The credentials I use to access my profile are no longer functional.",
        "intents": ["recover_password"],
        "difficulty": "very indirect",
        "note": "Cok resmi ve dolayli - recover password"
    },
    {
        "question": "What are my options if I want my item to arrive before the weekend but I also need to change the address?",
        "intents": ["delivery_options", "cancel_order"],
        "difficulty": "conditional + multi-intent",
        "note": "Kosullu soru + coklu intent"
    },
    {
        "question": "I regret making this purchase. How quickly can I reverse it and get my money back?",
        "intents": ["cancel_order"],
        "difficulty": "emotional + multi-step",
        "note": "Duygusal ifade + iptal ve para iadesi birlikte"
    },
    {
        "question": "Someone else might have accessed my profile. I want to secure it and also cancel any recent orders they may have placed.",
        "intents": ["recover_password", "cancel_order"],
        "difficulty": "security scenario",
        "note": "Guvenlik senaryosu - cok dolayli"
    },
    {
        "question": "i messed up my login n now i cant see if my stuff is coming tmrw or not n idk what 2 do",
        "intents": ["recover_password", "delivery_options"],
        "difficulty": "slang + multi-intent",
        "note": "Tam slang + coklu intent"
    },
    {
        "question": "If I decide not to keep my order and my account gets deleted, will I still get my refund?",
        "intents": ["cancel_order", "recover_password"],
        "difficulty": "hypothetical + multi-intent",
        "note": "Varsayimsal soru + coklu kavram"
    },
]

def run_hybrid_test():
    chain, retriever = load_rag()
    results = []

    print("=" * 60)
    print("ASAMA 3: HIBRIT / ZOORLAYICI TEST")
    print("Coklu intent ve dolayli sorular test ediliyor...")
    print("=" * 60)

    for i, item in enumerate(hybrid_questions):
        question   = item["question"]
        intents    = ", ".join(item["intents"])
        difficulty = item["difficulty"]
        note       = item["note"]

        rag_answer = chain.invoke(question)
        sources    = retriever.invoke(question)
        source_doc = sources[0].metadata["source"] if sources else "N/A"
        failed     = "could not find" in rag_answer.lower()
        status     = "BASARISIZ" if failed else "BASARILI"

        print(f"\n[{i+1}] {status}")
        print(f"Zorluk      : {difficulty}")
        print(f"Not         : {note}")
        print(f"Intents     : {intents}")
        print(f"Soru        : {question}")
        print(f"Cevap       : {rag_answer[:150]}...")
        print(f"Kaynak      : {source_doc}")
        print("-" * 60)

        results.append({
            "test_type": "hybrid",
            "intents": intents,
            "difficulty": difficulty,
            "note": note,
            "question": question,
            "rag_answer": rag_answer,
            "source": source_doc,
            "failed": failed
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("phase2/results/hybrid_results.csv", index=False)

    total   = len(results_df)
    success = len(results_df[results_df["failed"] == False])
    fail    = len(results_df[results_df["failed"] == True])

    print("\n" + "=" * 60)
    print("HIBRIT TEST SONUCLARI")
    print("=" * 60)
    print(f"Toplam     : {total}")
    print(f"Basarili   : {success} (%{round(success/total*100,1)})")
    print(f"Basarisiz  : {fail} (%{round(fail/total*100,1)})")
    print("\nZorluk seviyesine gore:")
    for diff in results_df["difficulty"].unique():
        sub = results_df[results_df["difficulty"] == diff]
        s = len(sub[sub["failed"] == False])
        durum = "Basarili" if s > 0 else "Basarisiz"
        print(f"  {diff}: {durum}")
    print(f"\nKaydedildi: phase2/results/hybrid_results.csv")

if __name__ == "__main__":
    run_hybrid_test()