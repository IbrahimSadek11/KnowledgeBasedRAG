"""
Neo4j graph operations
"""
from langchain_community.graphs import Neo4jGraph
from ..config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE


def init_graph():
    """Initialize Neo4j graph connection"""
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )
    
    graph.refresh_schema()
    return graph


def execute_query(graph, query):
    """Execute a Cypher query and return results"""
    try:
        result = graph.query(query)
        return result
    except Exception as e:
        print(f"Error executing query: {e}")
        return []
