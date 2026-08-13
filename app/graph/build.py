"""Graph assembly (PRD Section 3 — single StateGraph, four operation paths sharing
Validator/Audit Writer/state schema/checkpointer, D-23). Topology only — see
app/graph/nodes/*.py for which of the 14 nodes are real vs. stubbed and which step
fills each stub in. No fan-out (D-17), no streaming (D-28), no HITL interrupts (D-18) —
all locked, v2-deferred — so this is a strict DAG with exactly the edges below.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes import (
    audit_writer,
    cache_checker,
    cache_writer,
    chunker_node,
    deleter,
    duplicate_checker,
    embedder_query,
    embedding_batcher,
    generator,
    quality_gate,
    ranker,
    retriever,
    storer,
    validator,
)
from app.graph.routing import (
    route_after_cache_checker,
    route_after_duplicate_checker,
    route_after_embedder_query,
    route_after_embedding_batcher,
    route_after_quality_gate,
    route_after_retriever,
    route_after_storer,
    route_after_validator,
)
from app.graph.state import GraphState

NODE_FUNCS = {
    "validator": validator,
    "cache_checker": cache_checker,
    "embedder_query": embedder_query,
    "retriever": retriever,
    "ranker": ranker,
    "quality_gate": quality_gate,
    "generator": generator,
    "cache_writer": cache_writer,
    "duplicate_checker": duplicate_checker,
    "chunker": chunker_node,
    "embedding_batcher": embedding_batcher,
    "storer": storer,
    "deleter": deleter,
    "audit_writer": audit_writer,
}


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)
    for name, fn in NODE_FUNCS.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "validator")
    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "cache_checker": "cache_checker",
            "duplicate_checker": "duplicate_checker",
            "deleter": "deleter",
            "chunker": "chunker",
            "audit_writer": "audit_writer",
        },
    )

    # Query path (Nodes 2-8)
    graph.add_conditional_edges(
        "cache_checker",
        route_after_cache_checker,
        {"audit_writer": "audit_writer", "embedder_query": "embedder_query"},
    )
    graph.add_conditional_edges(
        "embedder_query",
        route_after_embedder_query,
        {"retriever": "retriever", "audit_writer": "audit_writer"},
    )
    graph.add_conditional_edges(
        "retriever",
        route_after_retriever,
        {"ranker": "ranker", "audit_writer": "audit_writer"},
    )
    graph.add_edge("ranker", "quality_gate")
    graph.add_conditional_edges(
        "quality_gate",
        route_after_quality_gate,
        {"generator": "generator", "audit_writer": "audit_writer"},
    )
    graph.add_edge("generator", "cache_writer")
    graph.add_edge("cache_writer", "audit_writer")

    # Ingest / update path (Nodes 9-12)
    graph.add_conditional_edges(
        "duplicate_checker",
        route_after_duplicate_checker,
        {"chunker": "chunker", "audit_writer": "audit_writer"},
    )
    graph.add_edge("chunker", "embedding_batcher")
    graph.add_conditional_edges(
        "embedding_batcher",
        route_after_embedding_batcher,
        {"storer": "storer", "audit_writer": "audit_writer"},
    )
    graph.add_conditional_edges(
        "storer",
        route_after_storer,
        {"deleter": "deleter", "audit_writer": "audit_writer"},
    )

    # Delete path (Node 13) — update's old-document cleanup converges here too.
    graph.add_edge("deleter", "audit_writer")

    graph.add_edge("audit_writer", END)
    return graph


def compile_graph(checkpointer=None) -> CompiledStateGraph:
    return build_graph().compile(checkpointer=checkpointer)
