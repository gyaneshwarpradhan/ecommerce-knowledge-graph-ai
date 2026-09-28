"""
tests/test_retrieval.py

Pytest tests for the Knowledge Graph and Retrieval layer.
No OpenAI API key required — LLM calls are mocked where needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Path helpers ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "ecommerce_data.json"

# ── Import project modules ────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph import KnowledgeGraph
from src.llm import validate_query
from src.retriever import Retriever


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def kg() -> KnowledgeGraph:
    return KnowledgeGraph(DATA_PATH)


@pytest.fixture(scope="module")
def retriever(kg: KnowledgeGraph) -> Retriever:
    return Retriever(kg)


# ── Graph Creation Tests ──────────────────────────────────────────────────────

class TestGraphCreation:
    def test_graph_loads_successfully(self, kg):
        assert kg.graph is not None

    def test_graph_has_nodes(self, kg):
        assert kg.graph.number_of_nodes() > 0

    def test_graph_has_edges(self, kg):
        assert kg.graph.number_of_edges() > 0

    def test_graph_has_customers(self, kg):
        customers = kg.get_nodes_by_type("Customer")
        assert len(customers) >= 5

    def test_graph_has_products(self, kg):
        products = kg.get_nodes_by_type("Product")
        assert len(products) >= 8

    def test_graph_has_brands(self, kg):
        brands = kg.get_nodes_by_type("Brand")
        assert len(brands) >= 3

    def test_graph_has_categories(self, kg):
        categories = kg.get_nodes_by_type("Category")
        assert len(categories) >= 3

    def test_graph_has_vendors(self, kg):
        vendors = kg.get_nodes_by_type("Vendor")
        assert len(vendors) >= 3

    def test_graph_has_orders(self, kg):
        orders = kg.get_nodes_by_type("Order")
        assert len(orders) >= 8

    def test_graph_has_special_values(self, kg):
        special = kg.get_nodes_by_type("SpecialValue")
        assert len(special) == 5


# ── Relationship Tests ────────────────────────────────────────────────────────

class TestRelationships:
    def test_product_has_brand(self, kg):
        """P001 (Air Max 270) should be MADE_BY B001 (Nike)."""
        brand_nodes = kg.get_neighbors("P001", relationship="MADE_BY")
        assert len(brand_nodes) == 1
        assert brand_nodes[0]["name"] == "Nike"

    def test_product_has_category(self, kg):
        """P001 should BELONG_TO CAT001 (Shoes)."""
        cat_nodes = kg.get_neighbors("P001", relationship="BELONGS_TO")
        assert len(cat_nodes) == 1
        assert cat_nodes[0]["name"] == "Shoes"

    def test_product_has_vendor(self, kg):
        """P001 should be SUPPLIED_BY V001 (ABC Sports)."""
        vendor_nodes = kg.get_neighbors("P001", relationship="SUPPLIED_BY")
        assert len(vendor_nodes) == 1
        assert vendor_nodes[0]["name"] == "ABC Sports"

    def test_customer_placed_order(self, kg):
        """C001 (Rahul Sharma) should have placed at least one order."""
        orders = kg.get_neighbors("C001", relationship="PLACED")
        assert len(orders) >= 1

    def test_order_contains_product(self, kg):
        """O001 should CONTAIN at least one product."""
        products = kg.get_neighbors("O001", relationship="CONTAINS")
        assert len(products) >= 1


# ── Retrieval Tests ───────────────────────────────────────────────────────────

class TestProductsByBrand:
    def test_nike_products(self, retriever):
        results = retriever.products_by_brand("Nike")
        assert len(results) > 0
        for r in results:
            assert r["brand"] == "Nike"

    def test_samsung_products(self, retriever):
        results = retriever.products_by_brand("Samsung")
        assert len(results) > 0
        for r in results:
            assert r["brand"] == "Samsung"

    def test_unknown_brand_returns_empty(self, retriever):
        results = retriever.products_by_brand("UnknownBrand999")
        assert results == []

    def test_brand_case_insensitive(self, retriever):
        results = retriever.products_by_brand("nike")
        assert len(results) > 0


class TestProductsByVendor:
    def test_abc_sports_products(self, retriever):
        results = retriever.products_by_vendor("ABC Sports")
        assert len(results) > 0
        for r in results:
            assert r["vendor"] == "ABC Sports"

    def test_tech_world_products(self, retriever):
        results = retriever.products_by_vendor("Tech World")
        assert len(results) > 0
        for r in results:
            assert r["vendor"] == "Tech World"

    def test_unknown_vendor_returns_empty(self, retriever):
        results = retriever.products_by_vendor("NoVendorHere")
        assert results == []


class TestProductsByBrandVendor:
    def test_nike_abc_sports(self, retriever):
        results = retriever.products_by_brand_vendor("Nike", "ABC Sports")
        assert len(results) > 0
        for r in results:
            assert r["brand"] == "Nike"
            assert r["vendor"] == "ABC Sports"

    def test_mismatched_brand_vendor_returns_empty(self, retriever):
        # Nike products are NOT supplied by Tech World
        results = retriever.products_by_brand_vendor("Nike", "Tech World")
        assert results == []

    def test_apple_tech_world(self, retriever):
        results = retriever.products_by_brand_vendor("Apple", "Tech World")
        assert len(results) > 0
        for r in results:
            assert r["brand"] == "Apple"
            assert r["vendor"] == "Tech World"


class TestProductsByCategory:
    def test_shoes_category(self, retriever):
        results = retriever.products_by_category("Shoes")
        assert len(results) > 0
        for r in results:
            assert r["category"] == "Shoes"

    def test_electronics_category(self, retriever):
        results = retriever.products_by_category("Electronics")
        assert len(results) > 0

    def test_unknown_category_returns_empty(self, retriever):
        results = retriever.products_by_category("Furniture")
        assert results == []


class TestCustomerOrders:
    def test_rahul_sharma_orders(self, retriever):
        results = retriever.customer_orders("Rahul Sharma")
        assert len(results) > 0
        for order in results:
            assert "order_id" in order
            assert "products" in order

    def test_unknown_customer_returns_empty(self, retriever):
        results = retriever.customer_orders("John Doe Fictional")
        assert results == []

    def test_orders_contain_products(self, retriever):
        results = retriever.customer_orders("Arjun Patel")
        assert len(results) > 0
        for order in results:
            assert len(order["products"]) > 0


class TestProductDetails:
    def test_air_max_details(self, retriever):
        result = retriever.product_details("Air Max 270")
        assert result["product"] == "Air Max 270"
        assert result["brand"] == "Nike"
        assert result["price"] == 12000

    def test_unknown_product_returns_empty(self, retriever):
        result = retriever.product_details("iPhone 99")
        assert result == {}

    def test_product_details_case_insensitive(self, retriever):
        result = retriever.product_details("air max 270")
        assert result.get("product") == "Air Max 270"


# ── Banana Tests ──────────────────────────────────────────────────────────────

class TestBananaRequirement:
    def test_banana_count_exactly_five(self, kg):
        """CRITICAL: Exactly 5 banana values must exist in the graph."""
        count = kg.validate_banana_count()
        assert count == 5, f"Expected 5 banana values, found {count}"

    def test_banana_search_returns_five(self, retriever):
        """search_value('banana') must return exactly 5 results."""
        results = retriever.search_value("banana")
        assert len(results) == 5, f"Expected 5 banana results, found {len(results)}"

    def test_banana_nodes_have_correct_type(self, retriever):
        """All banana nodes must be of type 'SpecialValue'."""
        results = retriever.search_value("banana")
        for node in results:
            assert node["type"] == "SpecialValue"

    def test_banana_node_ids(self, retriever):
        """Banana nodes should have ids SV001–SV005."""
        results = retriever.search_value("banana")
        ids = {r["id"] for r in results}
        expected = {"SV001", "SV002", "SV003", "SV004", "SV005"}
        assert ids == expected

    def test_banana_case_insensitive(self, retriever):
        """search_value should be case-insensitive."""
        results = retriever.search_value("BANANA")
        assert len(results) == 5

    def test_no_extra_banana_nodes(self, kg):
        """Ensure no more than 5 banana values exist anywhere in the graph."""
        banana_nodes = [
            nid
            for nid, attrs in kg.graph.nodes(data=True)
            if attrs.get("value", "").lower() == "banana"
        ]
        assert len(banana_nodes) == 5

    def test_banana_through_dispatcher(self, retriever):
        """Banana should be retrievable through the unified execute() dispatcher."""
        result = retriever.execute({"operation": "search_value", "value": "banana"})
        assert result is not None
        assert len(result) == 5


# ── Query Validation Tests ────────────────────────────────────────────────────

class TestQueryValidation:
    def test_valid_products_by_brand(self):
        is_valid, _ = validate_query({"operation": "products_by_brand", "brand": "Nike"})
        assert is_valid

    def test_valid_products_by_vendor(self):
        is_valid, _ = validate_query({"operation": "products_by_vendor", "vendor": "ABC Sports"})
        assert is_valid

    def test_valid_products_by_brand_vendor(self):
        is_valid, _ = validate_query(
            {"operation": "products_by_brand_vendor", "brand": "Nike", "vendor": "ABC Sports"}
        )
        assert is_valid

    def test_valid_customer_orders(self):
        is_valid, _ = validate_query({"operation": "customer_orders", "customer": "Rahul Sharma"})
        assert is_valid

    def test_valid_search_value(self):
        is_valid, _ = validate_query({"operation": "search_value", "value": "banana"})
        assert is_valid

    def test_valid_unsupported(self):
        is_valid, _ = validate_query({"operation": "unsupported"})
        assert is_valid

    def test_invalid_unknown_operation(self):
        is_valid, error = validate_query({"operation": "drop_all_tables"})
        assert not is_valid
        assert "Unknown operation" in error

    def test_invalid_missing_required_field(self):
        is_valid, error = validate_query({"operation": "products_by_brand"})
        assert not is_valid
        assert "missing required fields" in error

    def test_invalid_empty_query(self):
        is_valid, error = validate_query({})
        assert not is_valid

    def test_invalid_code_injection(self):
        is_valid, _ = validate_query({"operation": "exec('import os; os.system(\"rm -rf /\")')"})
        assert not is_valid


# ── Dispatcher Tests ──────────────────────────────────────────────────────────

class TestDispatcher:
    def test_unsupported_operation_returns_none(self, retriever):
        result = retriever.execute({"operation": "unsupported"})
        assert result is None

    def test_unknown_operation_returns_none(self, retriever):
        result = retriever.execute({"operation": "totally_made_up"})
        assert result is None

    def test_products_by_brand_via_dispatcher(self, retriever):
        result = retriever.execute({"operation": "products_by_brand", "brand": "Nike"})
        assert isinstance(result, list)
        assert len(result) > 0

    def test_customer_orders_via_dispatcher(self, retriever):
        result = retriever.execute(
            {"operation": "customer_orders", "customer": "Priya Mehta"}
        )
        assert isinstance(result, list)
        assert len(result) > 0


# ── Anti-Hallucination Tests ──────────────────────────────────────────────────

class TestAntiHallucination:
    def test_nonexistent_product_returns_empty(self, retriever):
        result = retriever.product_details("iPhone 99")
        assert result == {}

    def test_nonexistent_brand_returns_empty(self, retriever):
        result = retriever.products_by_brand("FakeBrand")
        assert result == []

    def test_nonexistent_customer_returns_empty(self, retriever):
        result = retriever.customer_orders("Nobody Real")
        assert result == []
