import numpy as np
import pandas as pd
from itertools import combinations
from typing import Optional, List, Tuple
import pulp as pl

def solve_design_network(
    balances: list[int],
    packages: list[tuple[int, float, float]],
    distance: float,
    connections: Optional[List[Tuple[int, int]]] = None,
):
    """
    balances  : list of integers  b(i)  ( + = supply, - = demand )
    packages  : [(type_id, capacity, infra_cost_per_unit_distance), ...]
                └➔ same for every undirected edge
    distance  : uniform edge length  m   (float)

    returns (status, opt_cost, chosen_packages, flows)
      chosen_packages[(u,v)] = [type_id, ...]   (may be empty)
      flows[(u,v)]           = flow from u→v    (float, signless)
    """

    n = len(balances)
    V = range(n)

    E = connections

    if E is None:
        E = [(u, v) for u, v in combinations(V, 2)]   # undirected edges

    # -------------------------------------------------- model
    mdl = pl.LpProblem("NetworkDesignMIP", pl.LpMinimize)

    # ---- 1. decision vars
    # y[e,k] binary  – buy package k on edge e?
    y = {
        (e, k): pl.LpVariable(f"y_{e}_{k}", 0, 1, pl.LpBinary)
        for e in E
        for k, _, _ in packages
    }

    # flow variables  f_{u,v}  (directed, non-negative)
    f = {}
    if connections is None:
        # | means to merge two dictionaries
        f = {
            (u, v): pl.LpVariable(f"f_{u}_{v}", lowBound=0)
            for u, v in E
        } | {
            (v, u): pl.LpVariable(f"f_{v}_{u}", lowBound=0)
            for u, v in E
        }
    else:
        for (u, v) in E:                        # use the pre-computed list
            f[(u, v)] = pl.LpVariable(f"f_{u}_{v}", lowBound=0)
            f[(v, u)] = pl.LpVariable(f"f_{v}_{u}", lowBound=0)

    # ---- 2. capacity linking
    for (u, v) in E:
        cap_expr = pl.lpSum(cap * y[((u, v), k)]
                            for k, cap, _ in packages)
        mdl += f[(u, v)] <= cap_expr
        mdl += f[(v, u)] <= cap_expr      # same total cap in opposite dir

    # ---- 3. flow conservation
    for i in V:
        inflow  = pl.lpSum(f[(j, i)] for j in V if (j, i) in f)
        outflow = pl.lpSum(f[(i, j)] for j in V if (i, j) in f)
        mdl += outflow - inflow == balances[i]

    # ---- 4. objective  (infrastructure only; add variable cost if needed)
    infra_cost = pl.lpSum(
        distance * cost * y[(e, k)]
        for e in E
        for k, _, cost in packages
    )
    mdl += infra_cost

    # -------------------------------------------------- solve
    status = mdl.solve(pl.PULP_CBC_CMD(msg=False))
    if status != pl.LpStatusOptimal:
        return pl.LpStatus[status], None, None, None

    # -------------------------------------------------- unpack solution
    chosen = {
        e: [k for k, _, _ in packages if pl.value(y[(e, k)]) > 0.5]
        for e in E
    }
    flow_sol = {edge: pl.value(var) for edge, var in f.items()}
    opt_cost = pl.value(mdl.objective)

    return "Optimal", opt_cost, chosen, flow_sol