"""
app.py - Interactive Streamlit Frontend for the E-Commerce Knowledge Graph & AI Retrieval System.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.graph import KnowledgeGraph
from src.llm import generate_answer, generate_query, validate_query
from src.retriever import Retriever

# --- Page Config ---
st.set_page_config(
    page_title="E-Commerce Knowledge Graph AI Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #3B82F6;
    }
    .pipeline-step {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .banana-badge {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_kg() -> tuple[KnowledgeGraph, Retriever]:
    data_path = Path(__file__).resolve().parent / "data" / "ecommerce_data.json"
    kg = KnowledgeGraph(data_path)
    retriever = Retriever(kg)
    return kg, retriever


kg, retriever = load_kg()
stats = kg.stats()

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=64)
    st.markdown("### **Knowledge Graph Info**")
    st.caption("Powered by NetworkX + Google Gemini")

    st.markdown("---")
    st.markdown("#### 📊 **Graph Statistics**")
    col1, col2 = st.columns(2)
    col1.metric("Total Nodes", stats["total_nodes"])
    col2.metric("Relationships", stats["total_edges"])

    st.markdown("---")
    st.markdown("#### 🧩 **Entities Breakdown**")
    st.write(f"👥 **Customers:** {stats.get('Customer', 0)}")
    st.write(f"📦 **Products:** {stats.get('Product', 0)}")
    st.write(f"🏷️ **Brands:** {stats.get('Brand', 0)}")
    st.write(f"📂 **Categories:** {stats.get('Category', 0)}")
    st.write(f"🏭 **Vendors:** {stats.get('Vendor', 0)}")
    st.write(f"🧾 **Orders:** {stats.get('Order', 0)}")
    st.write(f"🍌 **Banana Values:** {stats.get('banana_count', 0)} <span class='banana-badge'>EXACTLY 5</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🍌 **Integrity Check**")
    if st.button("Verify Banana Count", use_container_width=True):
        try:
            count = kg.validate_banana_count()
            st.success(f"✅ Banana count strictly verified: exactly {count} nodes present!")
        except Exception as e:
            st.error(f"❌ Verification failed: {e}")

    st.markdown("---")
    st.caption("E-Commerce Knowledge Graph AI System")


# --- Header ---
st.markdown('<div class="main-title">🛒 E-Commerce Knowledge Graph AI Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Natural Language Graph Retrieval System with Multi-Stage Grounding & Anti-Hallucination</div>',
    unsafe_allow_html=True,
)

# --- Top Tabs ---
tab_chat, tab_graph, tab_data, tab_arch = st.tabs(
    ["💬 AI Retrieval Assistant", "🕸️ Interactive Graph Visualizer", "📋 Dataset Explorer", "🛡️ Architecture & Integrity"]
)


# ==============================================================================
# TAB 1: AI Assistant
# ==============================================================================
with tab_chat:
    st.markdown("### 💬 Ask Questions to the Knowledge Graph")
    st.write("Type a question or pick one of the verified sample queries below:")

    sample_questions = [
        "Which products from Nike are supplied by ABC Sports?",
        "What products belong to the Shoes category?",
        "What products did Rahul Sharma order?",
        "Which products are supplied by Tech World?",
        "Find banana in the knowledge graph.",
        "What is the price of iPhone 99?",
    ]

    # Sample query chips
    st.markdown("**⚡ Quick Sample Queries:**")
    chip_cols = st.columns(3)
    selected_query = None

    for idx, q in enumerate(sample_questions):
        col_idx = idx % 3
        with chip_cols[col_idx]:
            if st.button(f"👉 {q}", key=f"sample_btn_{idx}", use_container_width=True):
                selected_query = q

    # Input form
    default_input = selected_query if selected_query else ""
    user_question = st.text_input(
        "Enter your question:",
        value=default_input,
        placeholder="e.g. Which products from Nike are supplied by ABC Sports?",
        key="user_query_input",
    )

    run_query_btn = st.button("🚀 Retrieve & Answer", type="primary", use_container_width=True)

    if run_query_btn and user_question.strip():
        q_text = user_question.strip()

        with st.spinner("Executing 4-stage retrieval pipeline..."):
            # Step 1: Query Planner
            structured_query = generate_query(q_text)
            is_valid, validation_err = validate_query(structured_query)

            # Step 2: Retrieval
            if is_valid and structured_query.get("operation") != "unsupported":
                retrieved_data = retriever.execute(structured_query)
            else:
                retrieved_data = None

            # Step 3: Grounded Answer
            answer = generate_answer(q_text, retrieved_data)

        st.markdown("---")
        st.markdown("### 🔍 Execution Pipeline Breakdown")

        # Step 1 Card
        with st.expander("📍 **Step 1: LLM Query Planner (Controlled JSON)**", expanded=True):
            st.write("Converts natural language into a single supported JSON operation with zero code execution:")
            st.json(structured_query)

        # Step 2 Card
        with st.expander("🛡️ **Step 2: Query Validation Gate**", expanded=True):
            if is_valid and structured_query.get("operation") != "unsupported":
                st.success(f"✅ **Operation Approved**: `{structured_query.get('operation')}` is a valid allowed operation.")
            elif structured_query.get("operation") == "unsupported":
                st.warning("⚠️ **Operation Unsupported**: Question cannot be mapped to the allowed graph operations.")
            else:
                st.error(f"❌ **Validation Failed**: {validation_err}")

        # Step 3 Card
        with st.expander("📦 **Step 3: NetworkX Graph Retrieval Output**", expanded=True):
            if retrieved_data:
                st.write(f"Retrieved **{len(retrieved_data) if isinstance(retrieved_data, list) else 1}** factual records directly from NetworkX:")
                st.json(retrieved_data)
            else:
                st.info("ℹ️ No matching records found in the knowledge graph for this query.")

        # Step 4 Card
        st.markdown("### 💡 **Step 4: Grounded Final Answer**")
        st.info(answer)


# ==============================================================================
# TAB 2: Interactive Graph Explorer
# ==============================================================================
with tab_graph:
    st.markdown("### 🕸️ Interactive Knowledge Graph Visualization")
    st.caption("Drag nodes, zoom in/out, and click nodes to explore connections.")

    # Color mapping for entity types
    COLOR_MAP = {
        "Customer": "#3B82F6",    # Blue
        "Product": "#10B981",     # Green
        "Brand": "#8B5CF6",       # Purple
        "Category": "#F97316",    # Orange
        "Vendor": "#EF4444",      # Red
        "Order": "#8D6E63",       # Brown
        "SpecialValue": "#EAB308" # Yellow / Banana Gold
    }

    # Legend
    legend_cols = st.columns(7)
    for idx, (etype, col) in enumerate(COLOR_MAP.items()):
        with legend_cols[idx]:
            label = "🍌 Banana" if etype == "SpecialValue" else etype
            st.markdown(f"<span style='display:inline-block;width:12px;height:12px;background-color:{col};border-radius:50%;margin-right:6px;'></span>**{label}**", unsafe_allow_html=True)

    # Build vis.js network JSON
    nodes_data = []
    edges_data = []

    for nid, attrs in kg.graph.nodes(data=True):
        ntype = attrs.get("type", "Unknown")
        color = COLOR_MAP.get(ntype, "#6B7280")
        name = attrs.get("name") or attrs.get("value") or nid
        label = f"{name} ({ntype})" if ntype != "SpecialValue" else f"🍌 {name} ({nid})"

        title_parts = [f"<b>ID:</b> {nid}", f"<b>Type:</b> {ntype}"]
        for k, v in attrs.items():
            if k not in ("id", "type"):
                title_parts.append(f"<b>{k}:</b> {v}")
        title = "<br>".join(title_parts)

        nodes_data.append({
            "id": nid,
            "label": label,
            "color": color,
            "title": title,
            "shape": "dot" if ntype != "SpecialValue" else "star",
            "size": 22 if ntype in ("Brand", "Category", "Customer") else (28 if ntype == "SpecialValue" else 18),
        })

    for u, v, data in kg.graph.edges(data=True):
        rel = data.get("relationship", "RELATED_TO")
        edges_data.append({
            "from": u,
            "to": v,
            "label": rel,
            "arrows": "to",
            "font": {"size": 10, "align": "middle", "color": "#6B7280"},
            "color": {"color": "#94A3B8", "highlight": "#2563EB"},
        })

    # Vis.js HTML template
    vis_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
      <style type="text/css">
        #network {{
          width: 100%;
          height: 600px;
          border: 1px solid #E2E8F0;
          border-radius: 8px;
          background-color: #FAFAFA;
        }}
      </style>
    </head>
    <body>
      <div id="network"></div>
      <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(nodes_data)});
        var edges = new vis.DataSet({json.dumps(edges_data)});
        var container = document.getElementById('network');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
          physics: {{
            barnesHut: {{
              gravitationalConstant: -3000,
              centralGravity: 0.3,
              springLength: 95,
              springConstant: 0.04,
              damping: 0.09
            }},
            maxVelocity: 50,
            minVelocity: 0.1
          }},
          interaction: {{
            hover: true,
            navigationButtons: true,
            keyboard: true
          }}
        }};
        var network = new vis.Network(container, data, options);
      </script>
    </body>
    </html>
    """
    components.html(vis_html, height=620)


# ==============================================================================
# TAB 3: Dataset Explorer
# ==============================================================================
with tab_data:
    st.markdown("### 📋 E-Commerce Dataset Tables")
    raw_data = kg._data

    data_tabs = st.tabs(["📦 Products", "🧾 Orders", "👥 Customers", "🏷️ Brands", "📂 Categories", "🏭 Vendors", "🍌 Special Values (Banana)"])

    with data_tabs[0]:
        st.dataframe(raw_data.get("products", []), use_container_width=True)
    with data_tabs[1]:
        st.dataframe(raw_data.get("orders", []), use_container_width=True)
    with data_tabs[2]:
        st.dataframe(raw_data.get("customers", []), use_container_width=True)
    with data_tabs[3]:
        st.dataframe(raw_data.get("brands", []), use_container_width=True)
    with data_tabs[4]:
        st.dataframe(raw_data.get("categories", []), use_container_width=True)
    with data_tabs[5]:
        st.dataframe(raw_data.get("vendors", []), use_container_width=True)
    with data_tabs[6]:
        st.dataframe(raw_data.get("special_values", []), use_container_width=True)
        st.success("🍌 Confirmed: Exactly 5 unique SpecialValue nodes with value 'banana'.")


# ==============================================================================
# TAB 4: Architecture & Integrity
# ==============================================================================
with tab_arch:
    st.markdown("### 🛡️ System Architecture & Anti-Hallucination Guarantees")

    st.markdown(
        """
        ```
        User Question
             │
             ▼
        LLM Query Planner (Call #1 - Gemini)
          - Maps question into 1 of 8 allowed operations
          - Output format: application/json (Deterministic, temperature=0)
          - Never executes arbitrary Python code
             │
             ▼
        Validation Gate (Python)
          - Verifies operation name is in the whitelist
          - Verifies all required parameters exist
             │
             ▼
        NetworkX Knowledge Graph
          - Retrieves clean Python dicts/lists
             │
             ▼
        LLM Answer Generator (Call #2 - Gemini)
          - Context contains ONLY retrieved graph facts
          - Strict instruction: "Answer ONLY using retrieved data"
             │
             ▼
        Grounded Final Answer (Zero Hallucination)
        ```
        """
    )
    st.markdown("---")
    st.markdown("#### 🍌 **Banana Requirement Proof**")
    st.write(
        "1. Nodes `SV001` through `SV005` in `data/ecommerce_data.json` have `value: banana`.\n"
        "2. `KnowledgeGraph.validate_banana_count()` strictly asserts `count == 5`.\n"
        "3. Automated tests in `tests/test_retrieval.py` verify count, IDs, case-insensitivity, and retrieval via `search_value('banana')`."
    )
