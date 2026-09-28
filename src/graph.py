"""
graph.py - Knowledge Graph implementation using NetworkX.

Loads the e-commerce dataset, builds a directed graph with typed nodes and
labelled edges, and exposes helper/validation methods.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx


class KnowledgeGraph:
    """Builds and manages an e-commerce knowledge graph with NetworkX."""

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)
        self.graph: nx.DiGraph = nx.DiGraph()
        self._data: dict[str, Any] = {}
        self._load_and_build()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_json(self) -> None:
        with open(self.data_path, "r", encoding="utf-8") as fh:
            self._data = json.load(fh)

    def _add_nodes(self) -> None:
        """Add all entity nodes to the graph."""

        for customer in self._data.get("customers", []):
            self.graph.add_node(
                customer["id"],
                type="Customer",
                name=customer["name"],
                email=customer["email"],
                city=customer["city"],
            )

        for brand in self._data.get("brands", []):
            self.graph.add_node(
                brand["id"],
                type="Brand",
                name=brand["name"],
                country=brand["country"],
            )

        for category in self._data.get("categories", []):
            self.graph.add_node(
                category["id"],
                type="Category",
                name=category["name"],
                description=category["description"],
            )

        for vendor in self._data.get("vendors", []):
            self.graph.add_node(
                vendor["id"],
                type="Vendor",
                name=vendor["name"],
                location=vendor["location"],
                rating=vendor["rating"],
            )

        for product in self._data.get("products", []):
            self.graph.add_node(
                product["id"],
                type="Product",
                name=product["name"],
                price=product["price"],
                stock=product["stock"],
            )

        for order in self._data.get("orders", []):
            self.graph.add_node(
                order["id"],
                type="Order",
                date=order["date"],
                status=order["status"],
                total=order["total"],
            )

        # Special banana nodes — exactly FIVE
        for sv in self._data.get("special_values", []):
            self.graph.add_node(
                sv["id"],
                type=sv["type"],           # "SpecialValue"
                value=sv["value"],         # "banana"
                description=sv["description"],
            )

    def _add_edges(self) -> None:
        """Add all relationship edges to the graph."""

        for product in self._data.get("products", []):
            pid = product["id"]

            # Product --MADE_BY--> Brand
            self.graph.add_edge(pid, product["brand_id"], relationship="MADE_BY")

            # Product --BELONGS_TO--> Category
            self.graph.add_edge(pid, product["category_id"], relationship="BELONGS_TO")

            # Product --SUPPLIED_BY--> Vendor
            self.graph.add_edge(pid, product["vendor_id"], relationship="SUPPLIED_BY")

        for order in self._data.get("orders", []):
            oid = order["id"]

            # Customer --PLACED--> Order
            self.graph.add_edge(order["customer_id"], oid, relationship="PLACED")

            # Order --CONTAINS--> Product
            for pid in order.get("product_ids", []):
                self.graph.add_edge(oid, pid, relationship="CONTAINS")

    def _load_and_build(self) -> None:
        self._load_json()
        self._add_nodes()
        self._add_edges()
        self.validate_graph_integrity()

    # ------------------------------------------------------------------ #
    # Public helpers                                                       #
    # ------------------------------------------------------------------ #

    def get_node(self, node_id: str) -> dict[str, Any]:
        """Return attributes of a single node."""
        if node_id not in self.graph:
            return {}
        return dict(self.graph.nodes[node_id])

    def get_nodes_by_type(self, node_type: str) -> list[dict[str, Any]]:
        """Return all nodes of a given type."""
        result = []
        for nid, attrs in self.graph.nodes(data=True):
            if attrs.get("type") == node_type:
                result.append({"id": nid, **attrs})
        return result

    def get_neighbors(
        self,
        node_id: str,
        relationship: str | None = None,
        direction: str = "out",
    ) -> list[dict[str, Any]]:
        """
        Return neighbors of *node_id*.

        direction: "out"  → successors (default)
                   "in"   → predecessors
        """
        if direction == "out":
            edges = self.graph.out_edges(node_id, data=True)
            neighbors = [(v, d) for _, v, d in edges]
        else:
            edges = self.graph.in_edges(node_id, data=True)
            neighbors = [(u, d) for u, _, d in edges]

        if relationship:
            neighbors = [(n, d) for n, d in neighbors if d.get("relationship") == relationship]

        return [{"id": n, **self.graph.nodes[n]} for n, _ in neighbors]

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def validate_graph_integrity(self) -> None:
        """Raise if the graph is missing expected node types or counts."""
        required_types = {"Customer", "Brand", "Category", "Vendor", "Product", "Order", "SpecialValue"}
        found_types = {attrs.get("type") for _, attrs in self.graph.nodes(data=True)}
        missing = required_types - found_types
        if missing:
            raise ValueError(f"Graph is missing node types: {missing}")

    def validate_banana_count(self) -> int:
        """Return the number of nodes with value == 'banana' and assert it equals 5."""
        banana_nodes = [
            nid
            for nid, attrs in self.graph.nodes(data=True)
            if attrs.get("value") == "banana"
        ]
        count = len(banana_nodes)
        if count != 5:
            raise ValueError(
                f"Expected exactly 5 banana values in the graph, found {count}."
            )
        return count

    # ------------------------------------------------------------------ #
    # Statistics                                                           #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, int]:
        """Return basic graph statistics."""
        type_counts: dict[str, int] = {}
        for _, attrs in self.graph.nodes(data=True):
            t = attrs.get("type", "Unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        banana_nodes = [
            nid
            for nid, attrs in self.graph.nodes(data=True)
            if attrs.get("value") == "banana"
        ]

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "banana_count": len(banana_nodes),
            **type_counts,
        }
