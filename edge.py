from typing import Optional


class Edge:
    """
    One directed arc in the residual network.
    `cost` may be modified in-place by Johnson potentials; `original_cost`
    preserves the input value for final cost accounting.
    """

    __slots__ = (
        "u",               # from node
        "v",               # to node
        "capacity",        # residual capacity
        "cost",            # *current* reduced cost
        "original_cost",   # keep the untouched input cost for calculating min cost
        "flow",            # current flow
        "reverse_edge",    # pointer to the opposite residual arc
    )

    def __init__(
        self,
        u: int,
        v: int,
        capacity: int,
        cost: int,
        reverse_edge: Optional["Edge"] = None, # "Edge" add quotes to refer to itself
    ) -> None:
        self.u = u
        self.v = v
        self.capacity = capacity
        self.cost = cost
        self.original_cost = cost
        self.flow = 0
        self.reverse_edge = reverse_edge  # will be set after both edges created

    # ------------------------------------------------------------------ helpers

    def is_reverse(self) -> bool:
        """
        True iff this edge is the zero-capacity backward partner.
        """
        return self.capacity == 0

    def remaining_capacity(self) -> int:
        return self.capacity - self.flow

    def augment(self, bottleneck: int) -> None:
        """
        Push `bottleneck` units through this residual arc.
        """
        self.flow += bottleneck
        self.reverse_edge.flow -= bottleneck

    # ------------------------------------------------------------------ dunder

    def __repr__(self) -> str:  # nice for debugging
        return (
            f"Edge({self.u}→{self.v}, cap={self.capacity}, "
            f"cost={self.original_cost}, flow={self.flow}, rev={self.is_reverse()})"
        )
