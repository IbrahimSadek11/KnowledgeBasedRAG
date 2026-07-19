"""
Explore the existing Neo4j knowledge graph's node properties.

Read-only exploration in preparation for building a separate Tabular RAG
pipeline. Reuses the existing Graph RAG Neo4j connection values from
backend/config.py (loaded from .env). Does not modify the graph.
"""
import os
import sys

from neo4j import GraphDatabase

# Add project root to path so we can reuse the existing connection config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE

print(f"Connexion a Neo4j: {NEO4J_URI} (database: {NEO4J_DATABASE})")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

try:
    with driver.session(database=NEO4J_DATABASE) as session:
        # 1. All node labels
        labels_result = session.run("CALL db.labels()")
        labels = [record["label"] for record in labels_result]

        print(f"\n=== Node labels ({len(labels)}) ===")
        for label in labels:
            print(f"  - {label}")

        # 2. One sample node per label, with all its properties
        for label in labels:
            print(f"\n=== Sample node for label: {label} ===")
            sample = session.run(
                f"MATCH (n:`{label}`) RETURN n LIMIT 1"
            ).single()

            if sample is None:
                print("  (no node found)")
                continue

            node = sample["n"]
            print(f"  element_id: {node.element_id}")
            print(f"  labels: {list(node.labels)}")
            print("  properties:")
            for key, value in dict(node).items():
                print(f"    {key}: {value!r}")
finally:
    driver.close()
    print("\nConnexion fermee.")
