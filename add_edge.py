from typing import List
from edge import Edge

def add_edge(graph: List[List[Edge]], u: int, v: int, cap: int, cost: int) -> None:
    """
    Append forward & reverse residual arcs to adjacency list (graph).
    """
    fwd = Edge(u, v, cap, cost)
    rev = Edge(v, u, 0, -cost)
    fwd.reverse_edge = rev
    rev.reverse_edge = fwd
    graph[u].append(fwd)
    graph[v].append(rev)