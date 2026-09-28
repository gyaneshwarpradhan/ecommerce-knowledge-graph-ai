"""
main.py - CLI entry point for the E-Commerce Knowledge Graph AI Assistant.

Flow per question:
  1. User Question
  2. LLM Query Planner  → Structured JSON Query (Gemini)
  3. Query Validation
  4. NetworkX Retrieval → Graph Data
  5. LLM Answer Generator → Grounded Answer (Gemini)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure UTF-8 stdout on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Allow running as `python src/main.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.graph import KnowledgeGraph
from src.llm import generate_answer, generate_query, validate_query
from src.retriever import Retriever

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ecommerce_data.json"


def print_separator(char: str = "-", width: int = 60) -> None:
    print(char * width)


def print_header() -> None:
    print_separator("=")
    print("  E-Commerce Knowledge Graph AI Assistant (Gemini Powered)")
    print_separator("=")


def display_stats(kg: KnowledgeGraph) -> None:
    stats = kg.stats()
    print(f"\n  Graph Statistics")
    print_separator()
    print(f"  Nodes            : {stats['total_nodes']}")
    print(f"  Relationships    : {stats['total_edges']}")
    print(f"  Customers        : {stats.get('Customer', 0)}")
    print(f"  Products         : {stats.get('Product', 0)}")
    print(f"  Brands           : {stats.get('Brand', 0)}")
    print(f"  Categories       : {stats.get('Category', 0)}")
    print(f"  Vendors          : {stats.get('Vendor', 0)}")
    print(f"  Orders           : {stats.get('Order', 0)}")
    print(f"  Banana values    : {stats.get('banana_count', 0)} [OK]")
    print_separator()


def handle_question(question: str, retriever: Retriever) -> None:
    """Run the full pipeline for one question and print the results."""
    print(f"\n[Question]: {question}")
    print_separator()

    # -- Step 1: LLM Query Planner (Gemini) --------------------------------
    print("[1] Generating structured query via LLM Query Planner...")
    structured_query = generate_query(question)

    print("\n[Structured Query]:")
    print(json.dumps(structured_query, indent=2))

    # -- Step 2: Validate the query ----------------------------------------
    is_valid, error = validate_query(structured_query)
    if not is_valid:
        print(f"\n[!] Query validation failed: {error}")
        print("\n[Answer]:\nI couldn't map your question to a supported operation.")
        print_separator()
        return

    operation = structured_query.get("operation", "")
    if operation == "unsupported":
        print("\n[Answer]:\nI couldn't find this information in the knowledge graph.")
        print_separator()
        return

    # -- Step 3: Retrieve graph data ---------------------------------------
    print("\n[2] Retrieving graph data from NetworkX...")
    retrieved_data = retriever.execute(structured_query)

    print("\n[Retrieved Data]:")
    if retrieved_data:
        print(json.dumps(retrieved_data, indent=2, ensure_ascii=False))
    else:
        print("  (no data found for this query)")

    # -- Step 4: LLM Answer Generator (Gemini) ----------------------------
    print("\n[3] Generating grounded answer via LLM Answer Generator...")
    answer = generate_answer(question, retrieved_data)

    print("\n[Answer]:")
    print(answer)
    print_separator()


def run_cli(kg: KnowledgeGraph) -> None:
    """Interactive CLI loop."""
    retriever = Retriever(kg)

    print('\nType your question and press Enter. Type "exit" to quit.\n')

    while True:
        try:
            question = input("Ask a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() == "exit":
            print("Goodbye!")
            break

        handle_question(question, retriever)


def main() -> None:
    print_header()

    # Load graph
    print("\nLoading knowledge graph...")
    try:
        kg = KnowledgeGraph(DATA_PATH)
    except Exception as exc:
        print(f"[Error] Failed to load graph: {exc}")
        sys.exit(1)

    print("[OK] Knowledge graph loaded successfully.")

    # Show stats
    display_stats(kg)

    # Validate banana count (will raise if not exactly 5)
    try:
        count = kg.validate_banana_count()
        print(f"Banana validation passed: exactly {count} banana values found.")
    except ValueError as exc:
        print(f"[!] Banana validation: {exc}")

    # Start interactive CLI
    run_cli(kg)


if __name__ == "__main__":
    main()
