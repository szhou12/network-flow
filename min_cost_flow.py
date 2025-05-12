from typing import List, Optional, Tuple
from edge import Edge
from add_edge import add_edge
from johnsons_reweighting import johnsons_reweighting
from get_augmenting_path import get_augmenting_path

INF = 10**18  # "∞" large enough for all problem sizes

def min_cost_max_flow_johnson(
    n: int,
    edges: List[List[int]],
    s: int,
    t: int,
    *,
    prebuilt: Optional[List[List[Edge]]] = None,
) -> Tuple[int, int, List[List[Edge]]]:
    """
    Min-Cost-Max-Flow using one Bellman-Ford pass for Johnson potentials,
    then Successive-Shortest-Path with Dijkstra.

    Parameters
    ----------
    n : int
        Number of vertices (0 … n-1).
    edges : List[[u,v,cap,cost]]
        Directed input arcs.  Multiple edges between same pair are allowed.
        used only when `prebuilt` is None
    s, t : int
        Source and sink vertices.
    prebuilt : (optional) already-constructed residual graph.
               If given, `edges` and `n` are ignored and the algorithm runs
               directly on this graph.

    Returns
    -------
    maxFlow, minCost, graph
    """
    # ---------------- build residual network ----------------
    if prebuilt is None:
        # -- build graph from `edges`
        graph: List[List[Edge]] = [[] for _ in range(n)]
        for u, v, cap, cost in edges:
            add_edge(graph, u, v, cap, cost)
    else:
        graph = prebuilt
        n = len(graph)           # update n so loops work correctly

    # ---------------- initial potentials via Bellman‑Ford ----------------
    dist = [INF] * n
    dist[s] = 0
    for _ in range(n - 1):
        updated = False
        for u in range(n):
            if dist[u] == INF:
                continue
            for e in graph[u]:
                if e.remaining_capacity() > 0 and dist[u] + e.cost < dist[e.v]:
                    dist[e.v] = dist[u] + e.cost
                    updated = True
        if not updated:  # early exit if nothing relaxed
            break

    # ---------------- Johnson re-weighting ----------------
    johnsons_reweighting(graph, dist)

    # ---------------- Successive‑Shortest‑Path ----------------
    max_flow = 0
    min_cost = 0
    while True:
        path, found = get_augmenting_path(graph, s, t)
        if not found:
            break

        # bottleneck on the path
        bottleneck = min(e.remaining_capacity() for e in path)

        # augment flow & accumulate real cost (use original_cost)
        for e in path:
            e.augment(bottleneck)
            min_cost += bottleneck * e.original_cost

        max_flow += bottleneck

    return max_flow, min_cost, graph


# ---------------------------------------------------------------------
# Min-Cost-Flow *with* arbitrary supplies/demands
# ---------------------------------------------------------------------
def min_cost_flow_with_balance(
    edges: List[List[int]],        # [u, v, capacity, cost]
    balance: List[int],            # + = supply, - = demand, len == n
) -> Tuple[int, int, List[List[Edge]]]:
    
    assert sum(balance) == 0, "total supply must equal total demand"

    n = len(balance)

    S, T = n, n + 1                # supersource, supersink
    N = n + 2
    graph: List[List[Edge]] = [[] for _ in range(N)]

    # (1) original network arcs
    for u, v, cap, cost in edges:
        add_edge(graph, u, v, cap, cost)

    # add adapater edges
    # (2) supply arcs  S -> v   and   demand arcs  v -> T
    total_supply = 0
    for v, b in enumerate(balance):
        if b > 0:                                    # warehouse
            add_edge(graph, S, v, b, 0)              # cost 0
            total_supply += b
        elif b < 0:                                  # store
            add_edge(graph, v, T, -b, 0)             # cost 0

    # (3) run the *existing* Johnson SSP on the augmented graph
    max_flow, min_cost, result_graph = min_cost_max_flow_johnson(N, [], S, T, prebuilt=graph)

    # (4) feasibility check
    if max_flow < total_supply:
        raise ValueError("no feasible shipping plan (capacities too small).")

    return max_flow, min_cost, result_graph