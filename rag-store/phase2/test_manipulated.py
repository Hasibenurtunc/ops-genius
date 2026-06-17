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

manipulations = [
    {
        "original": "How can I cancel my order?",
        "manipulated": "I want to abort my purchase.",
        "intent": "cancel_order",
        "manipulation_type": "synonym (cancel->abort, order->purchase)"
    },
    {
        "original": "I need to cancel my order.",
        "manipulated": "I'd like to undo my recent buy.",
        "intent": "cancel_order",
        "manipulation_type": "informal rephrasing"
    },
    {
        "original": "Can I cancel my order?",
        "manipulated": "Is it possible to stop my shipment before it leaves?",
        "intent": "cancel_order",
        "manipulation_type": "indirect phrasing"
    },
    {
        "original": "How do I cancel?",
        "manipulated": "pls cancel asap!!",
        "intent": "cancel_order",
        "manipulation_type": "slang + abbreviation"
    },
    {
        "original": "I want to cancel my order.",
        "manipulated": "I do NOT want this order anymore.",
        "intent": "cancel_order",
        "manipulation_type": "negation"
    },
    {
        "original": "What delivery options do you have?",
        "manipulated": "What shipping methods are available?",
        "intent": "delivery_options",
        "manipulation_type": "synonym (delivery->shipping)"
    },
    {
        "original": "How can I receive my package?",
        "manipulated": "What are my choices for getting my stuff delivered?",
        "intent": "delivery_options",
        "manipulation_type": "informal rephrasing"
    },
    {
        "original": "Tell me about delivery options.",
        "manipulated": "Do u guys do express or just normal shipping?",
        "intent": "delivery_options",
        "manipulation_type": "slang + specific question"
    },
    {
        "original": "What are your delivery options?",
        "manipulated": "I need my package fast, what are my options?",
        "intent": "delivery_options",
        "manipulation_type": "context added"
    },
    {
        "original": "How do you deliver orders?",
        "manipulated": "Can I choose how my order gets to me?",
        "intent": "delivery_options",
        "manipulation_type": "indirect phrasing"
    },
    {
        "original": "How do I reset my password?",
        "manipulated": "I forgot my login credentials, what do I do?",
        "intent": "recover_password",
        "manipulation_type": "synonym (password->credentials)"
    },
    {
        "original": "I can't log in, help me.",
        "manipulated": "I'm locked out of my account.",
        "intent": "recover_password",
        "manipulation_type": "indirect rephrasing"
    },
    {
        "original": "How to recover password?",
        "manipulated": "help i cant get into my acc!!",
        "intent": "recover_password",
        "manipulation_type": "slang + abbreviation"
    },
    {
        "original": "I need to reset my password.",
        "manipulated": "My password is not working anymore.",
        "intent": "recover_password",
        "manipulation_type": "problem description instead of request"
    },
    {
        "original": "Can you help me recover my account?",
        "manipulated": "I think someone hacked me and changed my password.",
        "intent": "recover_password",
        "manipulation_type": "added context/scenario"
    },
]

def run_manipulation_test():
    chain, retriever = load_rag()
    results = []

    print("=" * 60)
    print("ASAMA 2: MANIPULASYON TESTI")
    print("Kelime degisiklikleriyle sorular test ediliyor...")
    print("=" * 60)

    for i, item in enumerate(manipulations):
        original   = item["original"]
        manip_q    = item["manipulated"]
        intent     = item["intent"]
        manip_type = item["manipulation_type"]

        rag_answer = chain.invoke(manip_q)
        sources    = retriever.invoke(manip_q)
        source_doc = sources[0].metadata["source"] if sources else "N/A"
        failed     = "could not find" in rag_answer.lower()
        status     = "BASARISIZ" if failed else "BASARILI"

        print(f"\n[{i+1}] {status} | Intent: {intent}")
        print(f"Manipulasyon : {manip_type}")
        print(f"Orijinal     : {original}")
        print(f"Degistirilmis: {manip_q}")
        print(f"Cevap        : {rag_answer[:120]}...")
        print(f"Kaynak       : {source_doc}")
        print("-" * 60)

        results.append({
            "test_type": "manipulated",
            "intent": intent,
            "manipulation_type": manip_type,
            "original_question": original,
            "manipulated_question": manip_q,
            "rag_answer": rag_answer,
            "source": source_doc,
            "failed": failed
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("phase2/results/manipulated_results.csv", index=False)

    total   = len(results_df)
    success = len(results_df[results_df["failed"] == False])
    fail    = len(results_df[results_df["failed"] == True])

    print("\n" + "=" * 60)
    print("MANIPULASYON TEST SONUCLARI")
    print("=" * 60)
    print(f"Toplam     : {total}")
    print(f"Basarili   : {success} (%{round(success/total*100,1)})")
    print(f"Basarisiz  : {fail} (%{round(fail/total*100,1)})")
    print("\nManipulasyon tipine gore:")
    for mt in results_df["manipulation_type"].unique():
        sub = results_df[results_df["manipulation_type"] == mt]
        s = len(sub[sub["failed"] == False])
        durum = "Basarili" if s > 0 else "Basarisiz"
        print(f"  {mt}: {durum}")
    print(f"\nKaydedildi: phase2/results/manipulated_results.csv")

if __name__ == "__main__":
    run_manipulation_test()