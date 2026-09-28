# E-Commerce Knowledge Graph AI Retrieval System (Gemini Powered)

## Project Overview

This project implements a **Knowledge Graph** for an e-commerce dataset using **NetworkX** and an **AI-powered retrieval system** using the **Google Gemini API** (`google-genai` SDK). The system converts natural-language questions into structured graph queries, retrieves grounded data from the graph, and generates factual answers — with explicit safeguards against hallucination.

The application includes both a **Clean CLI interface** and a rich **Interactive Streamlit Web Dashboard** with real-time NetworkX graph visualization.

---

## Objective

Build a complete, runnable system that:

1. Represents e-commerce entities and relationships as a graph
2. Lets users ask natural-language questions
3. Converts those questions to controlled structured queries via Google Gemini
4. Retrieves facts exclusively from the graph
5. Generates grounded answers that cite only retrieved data
6. Provides an interactive UI for graph exploration and real-time execution inspection

---

## Architecture

```
User Question
     │
     ▼
LLM Query Planner (Call #1 - Gemini)
  - Converts NL question → JSON operation
  - Temperature = 0 (deterministic)
  - Returns only one of 8 supported operations
     │
     ▼
Query Validation
  - Checks operation name is in the allowed set
  - Checks required parameters are present
  - NEVER executes any LLM-generated code
     │
     ▼
NetworkX Knowledge Graph (Retriever)
  - Executes the validated structured query
  - Returns clean Python dicts/lists
  - No raw graph objects exposed
     │
     ▼
LLM Answer Generator (Call #2 - Gemini)
  - Receives ONLY: original question + retrieved data
  - Explicitly told NOT to invent any facts
  - Responds with "not found" if data is empty
     │
     ▼
Grounded Final Answer
```

---

## Technology Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| Graph database | NetworkX | Zero infrastructure, pure Python, fast in-memory traversal |
| LLM | Google Gemini (`gemini-3.6-flash`) | Fast, native JSON output, reliable free tier |
| LLM SDK | `google-genai` | Official modern Google GenAI SDK |
| Frontend | Streamlit + Vis.js | Real-time interactive UI and graph visualizer |
| Config | python-dotenv | Standard environment configuration |
| Tests | pytest | Standard Python test framework |
| Language | Python 3.10+ | Wide compatibility |

---

## Knowledge Graph Structure

### Entity Types (Nodes)

| Type | Count | Key Attributes |
|------|-------|----------------|
| Customer | 7 | id, name, email, city |
| Product | 12 | id, name, price, stock |
| Brand | 4 | id, name, country |
| Category | 4 | id, name, description |
| Vendor | 4 | id, name, location, rating |
| Order | 10 | id, date, status, total |
| SpecialValue | 5 | id, type, value="banana", description |

### Relationships (Edges)

| Source → Target | Relationship | Meaning |
|----------------|-------------|---------|
| Customer → Order | PLACED | Customer placed the order |
| Order → Product | CONTAINS | Order contains the product |
| Product → Brand | MADE_BY | Product is made by brand |
| Product → Category | BELONGS_TO | Product belongs to category |
| Product → Vendor | SUPPLIED_BY | Product is supplied by vendor |

---

## Retrieval Architecture

The `Retriever` class in `src/retriever.py` exposes these operations:

| Operation | Parameters | Description |
|-----------|-----------|-------------|
| `products_by_brand` | `brand` | All products from a brand |
| `products_by_vendor` | `vendor` | All products from a vendor |
| `products_by_brand_vendor` | `brand`, `vendor` | Products matching both brand AND vendor |
| `products_by_category` | `category` | All products in a category |
| `customer_orders` | `customer` | All orders for a customer with products |
| `product_details` | `product` | Full details for one product |
| `search_value` | `value` | All nodes with a matching `value` attribute |

---

## Banana Requirement

Exactly **5** nodes with `value == "banana"` exist in the graph:

```python
# In data/ecommerce_data.json
"special_values": [
    {"id": "SV001", "type": "SpecialValue", "value": "banana", ...},
    {"id": "SV002", "type": "SpecialValue", "value": "banana", ...},
    {"id": "SV003", "type": "SpecialValue", "value": "banana", ...},
    {"id": "SV004", "type": "SpecialValue", "value": "banana", ...},
    {"id": "SV005", "type": "SpecialValue", "value": "banana", ...}
]
```

These are retrievable via:
- `retriever.search_value("banana")` — returns all 5 nodes
- `kg.validate_banana_count()` — asserts count == 5 and raises ValueError if not
- Pytest test: `assert count == 5` in `tests/test_retrieval.py`

---

## Project Structure

```
ecommerce-knowledge-graph/
│
├── data/
│   └── ecommerce_data.json      # E-commerce dataset
│
├── src/
│   ├── __init__.py
│   ├── graph.py                 # KnowledgeGraph class (NetworkX)
│   ├── retriever.py             # Retriever class (all query operations)
│   ├── llm.py                   # LLM integration (Google Gemini SDK)
│   ├── prompts.py               # System prompts for both LLM calls
│   └── main.py                  # CLI entry point
│
├── app.py                       # Interactive Streamlit Web Frontend & Visualizer
├── tests/
│   └── test_retrieval.py        # pytest test suite (58 tests)
│
├── results/
│   └── sample_results.md        # Actual outputs from 6 sample questions
│
├── generate_sample_results.py   # Script to generate sample results
├── .env.example                 # Environment variable template
├── .gitignore                   # Excludes .env, .venv, __pycache__
├── requirements.txt             # Dependencies
└── README.md                    # Complete documentation
```

---

## Prerequisites

- Python 3.10 or higher
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/))

---

## Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd ecommerce-knowledge-graph

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## Environment Setup

```bash
# Copy the example file
cp .env.example .env

# Edit .env and set your key
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash
```

> **Never commit `.env`** — it is listed in `.gitignore`.

---

## Running the Applications

### 🌐 1. Interactive Web Frontend (Streamlit)
```bash
streamlit run app.py
```
This launches a web dashboard in your browser (`http://localhost:8501`) featuring:
- **AI Chat Assistant** with interactive query breakdown (Planner → Validation → NetworkX Facts → Grounded Answer)
- **Interactive Graph Visualizer** (Vis.js physics network with color-coded nodes and zoom/drag)
- **Dataset Explorer** (Browse products, orders, customers, and banana values in clean tables)
- **Banana Integrity Verification** button

### 💻 2. Interactive CLI Application
```bash
python src/main.py
```

---

## Running Tests

```bash
# Run all tests (no API key required)
pytest tests/ -v

# Run only banana tests
pytest tests/ -v -k "banana"

# Run only anti-hallucination tests
pytest tests/ -v -k "antihallu"
```
*(All 58 tests pass cleanly in ~0.8s)*

---

## Sample Questions

Try these in the Web UI or CLI:

1. `Which products from Nike are supplied by ABC Sports?`
2. `What products belong to the Shoes category?`
3. `What products did Rahul Sharma order?`
4. `Which products are supplied by Tech World?`
5. `Find banana in the knowledge graph.`
6. `What is the price of iPhone 99?` ← tests anti-hallucination
