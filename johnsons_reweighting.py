from typing import List
from edge import Edge

INF = 10**18  # "∞" large enough for all problem sizes

def johnsons_reweighting(graph: List[List[Edge]], dist: List[int]) -> None:
    """
    Johnson's reweighting algorithm:
        Reweight the edges of the graph to make all edge costs non-negative.
    """
    n = len(graph)

    for u in range(n):
        for e in graph[u]:
            if (
                e.remaining_capacity() > 0
                and dist[u] < INF and dist[e.v] < INF   # guard!
            ):
                e.cost += dist[u] - dist[e.v]
            else:
                e.cost = 0           # saturated arcs carry no residual cost