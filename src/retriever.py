"""
retriever.py - Structured retrieval layer over the KnowledgeGraph.

All public methods return plain Python dicts/lists — no raw NetworkX objects.
"""

from __future__ import annotations

from typing import Any

from src.graph import KnowledgeGraph


class Retriever:
    """Executes structured queries against the KnowledgeGraph."""

    def __init__(self, kg: KnowledgeGraph) -> None:
        self.kg = kg

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _node_name(self, node_id: str) -> str:
        return self.kg.get_node(node_id).get("name", node_id)

    def _product_detail(self, pid: str) -> dict[str, Any]:
        """Build a full product detail dict from the graph."""
        product = self.kg.get_node(pid)
        if not product:
            return {}

        brand_nodes = self.kg.get_neighbors(pid, relationship="MADE_BY")
        category_nodes = self.kg.get_neighbors(pid, relationship="BELONGS_TO")
        vendor_nodes = self.kg.get_neighbors(pid, relationship="SUPPLIED_BY")

        return {
            "id": pid,
            "product": product.get("name", ""),
            "price": product.get("price"),
            "stock": product.get("stock"),
            "brand": brand_nodes[0].get("name", "") if brand_nodes else "",
            "category": category_nodes[0].get("name", "") if category_nodes else "",
            "vendor": vendor_nodes[0].get("name", "") if vendor_nodes else "",
        }

    def _find_node_id_by_name(self, node_type: str, name: str) -> str | None:
        """Case-insensitive name lookup for a given node type."""
        name_lower = name.strip().lower()
        for nid, attrs in self.kg.graph.nodes(data=True):
            if attrs.get("type") == node_type and attrs.get("name", "").lower() == name_lower:
                return nid
        return None

    # ------------------------------------------------------------------ #
    # Public retrieval methods                                             #
    # ------------------------------------------------------------------ #

    def products_by_brand(self, brand: str) -> list[dict[str, Any]]:
        """Return all products made by *brand*."""
        brand_id = self._find_node_id_by_name("Brand", brand)
        if not brand_id:
            return []

        results = []
        for nid, attrs in self.kg.graph.nodes(data=True):
            if attrs.get("type") == "Product":
                brand_neighbors = self.kg.get_neighbors(nid, relationship="MADE_BY")
                if any(n["id"] == brand_id for n in brand_neighbors):
                    results.append(self._product_detail(nid))
        return results

    def products_by_vendor(self, vendor: str) -> list[dict[str, Any]]:
        """Return all products supplied by *vendor*."""
        vendor_id = self._find_node_id_by_name("Vendor", vendor)
        if not vendor_id:
            return []

        results = []
        for nid, attrs in self.kg.graph.nodes(data=True):
            if attrs.get("type") == "Product":
                vendor_neighbors = self.kg.get_neighbors(nid, relationship="SUPPLIED_BY")
                if any(n["id"] == vendor_id for n in vendor_neighbors):
                    results.append(self._product_detail(nid))
        return results

    def products_by_brand_vendor(self, brand: str, vendor: str) -> list[dict[str, Any]]:
        """Return products that match both *brand* AND *vendor*."""
        brand_results = {r["id"] for r in self.products_by_brand(brand)}
        vendor_results = {r["id"] for r in self.products_by_vendor(vendor)}
        common_ids = brand_results & vendor_results
        return [self._product_detail(pid) for pid in common_ids]

    def products_by_category(self, category: str) -> list[dict[str, Any]]:
        """Return all products in *category*."""
        cat_id = self._find_node_id_by_name("Category", category)
        if not cat_id:
            return []

        results = []
        for nid, attrs in self.kg.graph.nodes(data=True):
            if attrs.get("type") == "Product":
                cat_neighbors = self.kg.get_neighbors(nid, relationship="BELONGS_TO")
                if any(n["id"] == cat_id for n in cat_neighbors):
                    results.append(self._product_detail(nid))
        return results

    def customer_orders(self, customer: str) -> list[dict[str, Any]]:
        """Return all orders placed by *customer*, with their products."""
        cust_id = self._find_node_id_by_name("Customer", customer)
        if not cust_id:
            return []

        cust_name = self.kg.get_node(cust_id).get("name", customer)
        order_nodes = self.kg.get_neighbors(cust_id, relationship="PLACED")
        results = []
        for order in order_nodes:
            oid = order["id"]
            order_attrs = self.kg.get_node(oid)
            product_nodes = self.kg.get_neighbors(oid, relationship="CONTAINS")
            products = [self._product_detail(p["id"]) for p in product_nodes]
            results.append(
                {
                    "customer": cust_name,
                    "order_id": oid,
                    "date": order_attrs.get("date"),
                    "status": order_attrs.get("status"),
                    "total": order_attrs.get("total"),
                    "products": products,
                }
            )
        return results

    def product_details(self, product: str) -> dict[str, Any]:
        """Return full details for a single product by name."""
        product_lower = product.strip().lower()
        for nid, attrs in self.kg.graph.nodes(data=True):
            if attrs.get("type") == "Product" and attrs.get("name", "").lower() == product_lower:
                return self._product_detail(nid)
        return {}

    def search_value(self, value: str) -> list[dict[str, Any]]:
        """
        Return all nodes whose 'value' attribute equals *value* (case-insensitive).
        Used to retrieve banana nodes and any other SpecialValue nodes.
        """
        value_lower = value.strip().lower()
        results = []
        for nid, attrs in self.kg.graph.nodes(data=True):
            if attrs.get("value", "").lower() == value_lower:
                results.append({"id": nid, **attrs})
        return results

    # ------------------------------------------------------------------ #
    # Dispatcher — called by the main pipeline                            #
    # ------------------------------------------------------------------ #

    def execute(self, query: dict[str, Any]) -> Any:
        """
        Dispatch a structured query dict to the correct retrieval method.

        Returns the retrieved data or None if the operation is unsupported/invalid.
        """
        operation = query.get("operation", "")

        if operation == "products_by_brand":
            return self.products_by_brand(query.get("brand", ""))

        if operation == "products_by_vendor":
            return self.products_by_vendor(query.get("vendor", ""))

        if operation == "products_by_brand_vendor":
            return self.products_by_brand_vendor(
                query.get("brand", ""), query.get("vendor", "")
            )

        if operation == "products_by_category":
            return self.products_by_category(query.get("category", ""))

        if operation == "customer_orders":
            return self.customer_orders(query.get("customer", ""))

        if operation == "product_details":
            return self.product_details(query.get("product", ""))

        if operation == "search_value":
            return self.search_value(query.get("value", ""))

        # "unsupported" or anything else
        return None
