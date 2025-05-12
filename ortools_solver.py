from typing import List
import numpy as np
from ortools.graph.python import min_cost_flow

def ortools_solver(edges: List[List[int]], balance: List[int]) -> None:
    # Instantiate a SimpleMinCostFlow solver.
    smcf = min_cost_flow.SimpleMinCostFlow()


    # Define four parallel arrays: from-node, to-node, capacities, and unit costs.
    start_nodes = np.array([edge[0] for edge in edges])
    end_nodes = np.array([edge[1] for edge in edges])
    capacities = np.array([edge[2] for edge in edges])
    unit_costs = np.array([edge[3] for edge in edges])

    # Add arcs, capacities and costs in bulk using numpy.
    all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
        start_nodes, end_nodes, capacities, unit_costs
    )

    # Add supply for each nodes.
    smcf.set_nodes_supplies(np.arange(0, len(balance)), balance)

    # Find the min cost flow.
    status = smcf.solve()

    if status != smcf.OPTIMAL:
        print("There was an issue with the min cost flow input.")
        print(f"Status: {status}")
        exit(1)
    print(f"Minimum cost: {smcf.optimal_cost()}")
    print("")
    print(" Arc    Flow / Capacity Cost")
    solution_flows = smcf.flows(all_arcs)
    costs = solution_flows * unit_costs
    for arc, flow, cost in zip(all_arcs, solution_flows, costs):
        print(
            f"{smcf.tail(arc):1} -> "
            f"{smcf.head(arc)}  {flow:3}  / {smcf.capacity(arc):3}       {cost}"
        )

