# Ops-Genius — AI-Powered E-Commerce Customer Support System

> **Graduation Thesis Project** — Çukurova University, Department of Computer Engineering  
> Student: Hasibe Nur Tunç (2021556067) | Advisor: Prof. Dr. Umut Orhan

A thesis project implementing a **Retrieval-Augmented Generation (RAG)** pipeline embedded into a fully functional e-commerce storefront. The AI assistant answers customer questions about orders, refunds, delivery, and account management by grounding responses strictly in policy documents — eliminating hallucination and ensuring every answer is traceable to a source document.

---
## Project Showcase & Demo Video

###  Live System Walkthrough
>  *Below is a full interactive walkthrough demonstrating the FastAPI storefront, product filtering, shopping cart animations, and the integrated real-time RAG customer support widget.*

ss/demo.mp4

---

###  UI Screenshots

|  E-Commerce Main Storefront |  Integrated AI Assistant (RAG) |  Responsive Footer & Chat View |
|---|---|---|
| ![Storefront Landing](ss/ss1.png.jpg) | ![Chatbot Active](ss/ss2.png.jpg) | ![Store Footer View](ss/ss3.png.jpg) |

---

## Live Demo

**Option A — Standalone Chat Interface (Streamlit):**
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

**Option B — Full E-Commerce Store (FastAPI):**
```bash
python server.py
# Auto-opens at http://localhost:8000
```

---

## Architecture

```
User Question (via Chat Widget or Streamlit)
     │
     ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
     │
     ▼
ChromaDB Vector Store ──► Top-3 relevant policy chunks
     │
     ▼
Prompt Template + LLM (Llama 3.1 8B via Groq API)
     │
     ▼
Answer + Source Document (returned to UI)
```

**Two interfaces, one RAG backend:**
- `app.py` — Streamlit standalone chat interface
- `server.py` — FastAPI server powering the full e-commerce store + `/api/chat` endpoint

---

## Technology Stack

| Component | Technology |
|---|---|
| RAG Framework | LangChain |
| Language Model | `llama-3.1-8b-instant` via Groq API (free tier) |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database | ChromaDB (local, persisted) |
| Text Splitter | RecursiveCharacterTextSplitter (chunk=500, overlap=50) |
| Backend | FastAPI + Uvicorn |
| Standalone Chat UI | Streamlit |
| Store Frontend | Vanilla HTML + CSS + JavaScript |

---

## Project Structure

```
ops-genius/
├── server.py               # FastAPI: e-commerce store + /api/chat endpoint
├── app.py                  # Streamlit standalone chat interface
├── app_backup.py           # Streamlit backup (previous version)
├── rag_pipeline.py         # RAG pipeline builder (indexing + querying)
├── demo.py                 # Terminal demo
├── explore_data.py         # Dataset exploration notebook helper
├── test_with_bitext.py     # Full Bitext dataset integration test
├── analyze_results.py      # General result analysis
├── test_results.csv        # Phase 1 test results (54 queries)
│
├── docs/                   # Policy documents (knowledge base)
│   ├── order_policy.txt
│   ├── refund_policy.txt
│   ├── delivery_policy.txt
│   ├── account_policy.txt
│   └── support_policy.txt
│
├── static/                 # E-commerce store frontend
│   ├── index.html          # Main store page
│   ├── style.css           # Full responsive stylesheet
│   ├── script.js           # Product grid, cart, chat widget logic
│   └── images/             # Product images (.webp)
│       ├── headphones.webp
│       ├── smartwatch.webp
│       ├── laptop_stand.webp
│       ├── wallet.webp
│       ├── running_shoes.webp
│       ├── tshirt.webp
│       ├── backpack.webp
│       └── sunglasses.webp
│
├── chroma_db/              # Persisted vector store (auto-generated)
│
└── phase2/                 # Evaluation experiments
    ├── filter_dataset.py
    ├── filtered_dataset.csv
    ├── test_controlled.py
    ├── test_manipulated.py
    ├── test_hybrid.py
    ├── analyze_failed.py
    └── results/
        ├── controlled_results.csv
        ├── manipulated_results.csv
        └── hybrid_results.csv
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/yourusername/ops-genius.git
cd ops-genius

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install fastapi uvicorn streamlit langchain langchain-groq langchain-chroma \
            langchain-huggingface chromadb sentence-transformers python-dotenv pandas
```

### 2. Configure environment

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 3. Build the vector store (run once)

```bash
python rag_pipeline.py
```

---

## E-Commerce Store Features

The store is served by FastAPI and built with vanilla HTML/CSS/JS. No frontend framework required.

**Store (`/`):**
- 8 products across 3 categories: Electronics, Clothing, Accessories
- Category filter bar (All / Electronics / Clothing / Accessories)
- Add to cart with toast notification and animated button feedback
- Cart sidebar with item management and total calculation
- Responsive design (mobile, tablet, desktop)
- Sticky header with blur effect

**AI Chat Widget:**
- Floating button (bottom-right corner) opens/closes the chat
- Animated typing indicator while waiting for RAG response
- Quick action buttons: Cancel Order / Account Help / Delivery Info
- Source document badge displayed under each response
- Smooth open/close animation

**API Endpoints:**
```
GET  /api/products  →  Returns list of 8 products
POST /api/chat      →  Accepts {"question": "..."}, returns {"answer": "...", "source": "..."}
```

---

## Products Catalog

| # | Name | Category | Price |
|---|---|---|---|
| 1 | Wireless Headphones | Electronics | $79.99 |
| 2 | Smart Watch Pro | Electronics | $199.99 |
| 3 | Laptop Stand | Accessories | $49.99 |
| 4 | Leather Wallet | Accessories | $39.99 |
| 5 | Running Shoes | Clothing | $129.99 |
| 6 | Premium T-Shirt | Clothing | $29.99 |
| 7 | Urban Backpack | Accessories | $89.99 |
| 8 | Aviator Sunglasses | Accessories | $59.99 |

---

## Dataset

**Bitext Customer Support LLM Chatbot Training Dataset**  
Source: [HuggingFace](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)

- 26,872 real customer support interactions
- 27 intent categories
- 5 columns: `instruction`, `intent`, `category`, `response` (ground truth), `flags`

**3 intents selected for Phase 2 evaluation:**

| Intent | Category | Description |
|---|---|---|
| `cancel_order` | ORDER | Order cancellation requests |
| `recover_password` | ACCOUNT | Password and credential recovery |
| `delivery_options` | DELIVERY | Shipping method inquiries |

---

## Knowledge Base

Since the Bitext dataset does not include policy documents, a **synthetic knowledge base** was constructed referencing publicly available corporate policies (Amazon, PayPal, Shopify, FedEx).

| Document | Topics Covered |
|---|---|
| `order_policy.txt` | Placing, cancelling, modifying, tracking orders |
| `refund_policy.txt` | Refund eligibility, process, timelines |
| `delivery_policy.txt` | Shipping options, estimated times, international delivery |
| `account_policy.txt` | Registration, password/PIN recovery, account deletion |
| `support_policy.txt` | Support hours, complaint process, contact methods |

Questions outside these documents return: *"I could not find information about this in our policy documents."*

---

## Evaluation

### Phase 1 — Initial Baseline (54 queries, 27 intents)

| Metric | Value |
|---|---|
| Total Queries | 54 |
| Successful | 30 (%55.6) |
| Failed | 24 (%44.4) |

### Phase 2 — Three-Stage Evaluation (3 selected intents)

| Test Phase | Queries | Before | After |
|---|---|---|---|
| Controlled | 300 | 53.7% | **98.7%** |
| Manipulation | 15 | 60–66% | **86.7%** |
| Hybrid Stress | 10 | 90% | **100%** |

**Controlled test breakdown:**

| Intent | Before | After |
|---|---|---|
| cancel_order | 85% | 94% |
| delivery_options | 53% | 99% |
| recover_password | 23% | 95% |

### Failure Analysis & Solutions

| Failure Type | Root Cause | Solution |
|---|---|---|
| Placeholder Tokens | `{{Order Number}}`, `{{Delivery City}}` treated as literal text | Query preprocessing |
| Synonym Mismatch | "PIN code", "access key", "pwd" not in document vocabulary | Document enrichment |
| Indirect Phrasing | "stop my shipment" not matched to cancel policy | Prompt + document update |
| Informal Language | Slang, abbreviations, typos | Prompt instruction update |

---

## Key Design Decisions

- **Strict grounding:** Prompt instructs the LLM to answer *only* from retrieved context — no hallucination.
- **Informal language handling:** Prompt explicitly handles slang, abbreviations, and indirect phrasing.
- **Synonym-rich documents:** Policy documents include alternative terminology (PIN/password, shipping/delivery, purchase/order).
- **Chunk overlap:** `chunk_size=500`, `chunk_overlap=50` preserves context across boundaries.
- **Top-k retrieval:** `k=3` chunks per query balances relevance and context length.
- **Source attribution:** Every response includes the source document name for full traceability.
- **API-based LLM:** Groq API used instead of local model — hardware-agnostic, no GPU required.
- **Dual interface:** Same RAG backend powers both the Streamlit chat and FastAPI store.

---

## Running Tests

```bash
# Controlled test (300 queries)
python phase2/test_controlled.py

# Manipulation test (15 queries)
python phase2/test_manipulated.py

# Hybrid stress test (10 queries)
python phase2/test_hybrid.py

# Failure analysis
python phase2/analyze_failed.py

# General analysis
python analyze_results.py
```

---

## References

- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020.*
- Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey. *arXiv:2312.10997.*
- Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *arXiv:2309.15217.*
- Yao et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023.*
- Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019.*
- Bitext. (2024). Bitext Customer Support LLM Chatbot Training Dataset. HuggingFace.

---

## License

Developed as part of a graduation thesis at Çukurova University, Faculty of Engineering, Department of Computer Engineering. For academic use only.
