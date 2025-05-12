from typing import List, Tuple
import heapq

from edge import Edge
from johnsons_reweighting import johnsons_reweighting


INF = 10**18  # "∞" large enough for all problem sizes

def get_augmenting_path(
    graph: List[List[Edge]],
    s: int,
    t: int,
) -> Tuple[List[Edge], bool]:
    """
    Dijkstra over Johnson-reweighted residual costs.
    Returns (path_edges, found_flag).  If no s-t path exists, found_flag == False.
    """
    n = len(graph)
    dist = [INF] * n
    prev = [None] * n

    pq = []

    # start node
    heapq.heappush(pq, (0, s))

    # loop
    while pq:
        # current node: pop the top
        distance, u = heapq.heappop(pq)
        # check if visited
        if dist[u] != INF:
            continue
        # update
        dist[u] = distance

        # make the next move
        for e in graph[u]:
            v = e.v
            # check if edge has remaining cap
            if e.remaining_capacity() == 0:
                continue
            # check if visited
            if dist[v] != INF:
                continue

            heapq.heappush(pq, (distance + e.cost, v))
            prev[v] = e
    
    # Sink t not reachable
    if dist[t] == INF:
        return [], False
    
    # Johnson's reweighting again
    johnsons_reweighting(graph, dist)

    # reconstruct the path
    path = []
    v = t
    while v != s:
        e = prev[v]
        path.append(e)
        v = e.u
    path.reverse()

    return path, True


