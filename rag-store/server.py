import os
import webbrowser
import threading
# pyrefly: ignore [missing-import]
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI(title="Ops-Genius E-Commerce")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── RAG Pipeline ─────────────────────────────────────────
print("Loading RAG pipeline...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

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

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
print("RAG pipeline ready!")

# ── Products ─────────────────────────────────────────────
PRODUCTS = [
    {"id":1,"name":"Wireless Headphones","price":79.99,"category":"Electronics","image":"/static/images/headphones.webp","description":"Premium over-ear headphones with active noise cancellation and 30h battery."},
    {"id":2,"name":"Smart Watch Pro","price":199.99,"category":"Electronics","image":"/static/images/smartwatch.webp","description":"Advanced fitness tracking, heart-rate monitor and AMOLED display."},
    {"id":3,"name":"Laptop Stand","price":49.99,"category":"Accessories","image":"/static/images/laptop_stand.webp","description":"Ergonomic aluminum stand with adjustable height."},
    {"id":4,"name":"Leather Wallet","price":39.99,"category":"Accessories","image":"/static/images/wallet.webp","description":"Handcrafted genuine leather wallet with RFID blocking."},
    {"id":5,"name":"Running Shoes","price":129.99,"category":"Clothing","image":"/static/images/running_shoes.webp","description":"Lightweight performance shoes with responsive cushioning."},
    {"id":6,"name":"Premium T-Shirt","price":29.99,"category":"Clothing","image":"/static/images/tshirt.webp","description":"100% organic cotton crew-neck with modern relaxed fit."},
    {"id":7,"name":"Urban Backpack","price":89.99,"category":"Accessories","image":"/static/images/backpack.webp","description":"Water-resistant daypack with padded laptop compartment."},
    {"id":8,"name":"Aviator Sunglasses","price":59.99,"category":"Accessories","image":"/static/images/sunglasses.webp","description":"Classic aviator with polarised UV400 lenses."},
]

# ── Routes ───────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/api/products")
async def get_products():
    return JSONResponse(content=PRODUCTS)

@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    question = body.get("question", "")
    if not question:
        return JSONResponse(content={"answer": "Please ask a question.", "source": ""})
    try:
        answer = rag_chain.invoke(question)
        sources = retriever.invoke(question)
        source = sources[0].metadata["source"].replace(".txt", "") if sources else "N/A"
        return JSONResponse(content={"answer": answer, "source": source})
    except Exception as e:
        return JSONResponse(content={"answer": f"Sorry, an error occurred: {str(e)}", "source": ""}, status_code=500)

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:8000")).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
