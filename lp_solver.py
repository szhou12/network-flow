from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import time
from datetime import datetime
from itertools import combinations
import networkx as nx
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus


def construct_graph(*, nodes_df, edges_df=None, distance=200):
    # directed graph
    G = nx.DiGraph()

    for _, row in nodes_df.iterrows():
        # negate demand value for the convenience of flow conservation constraint: 
        # (inflow - outflow) == balance
        # supply node needs to satisfy (inflow - outflow) = -supply (negative)
        # demand node needs to satisfy (inflow - outflow) = demand (positive)
        G.add_node(row['省份编号'], balance = -row['供需量（万吨）'])

    # add directed edges in both directions instead of undirected edges
    # so that one flow variable corresponds to one directed edge
    # if edges_df is given, make sure each row is one-directional (u -> v). if u-v needs to be bi-directional, resolve it in input edges_df (add two rows: u->v and v->u). NOT HERE!
    if edges_df is not None:
        valid_edges = edges_df[edges_df['是否可连 (1/0)'] == 1]
        for _, row in valid_edges.iterrows():
            u, v, dist = row['起点省份编号'], row['终点省份编号'], row['距离 (km)']
            G.add_edge(u, v, distance=dist)
        #     G.add_edge(v, u, distance=dist)
        # G.add_edge(21, 20, distance=40)
        # G.add_edge(20, 21, distance=40)
    else:
        # fully-connected bi-directional graph
        n = nodes_df.shape[0]
        V = range(1, n + 1)
        E = [(u, v) for u, v in combinations(V, 2)]
        for u, v in E:
            G.add_edge(u, v, distance=distance)
            G.add_edge(v, u, distance=distance)
        
    return G


class Pipe:
    def __init__(self):
        self.pipe_costs = []
    
    @classmethod
    def from_dataframe(cls, pipeline_cost_df):
        """
        Create a Pipe object from input dataframe.
        """
        # validate dataframe containing required columns
        required_columns = ['流量范围（万吨）', '单位成本（万元/km）']
        if not all(col in pipeline_cost_df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        # Init Pipe object
        pipe = cls()

        # load data
        for _, row in pipeline_cost_df.iterrows():
            try:
                flow_range = row['流量范围（万吨）']
                if '-' in flow_range:
                    low, high = map(float, flow_range.split('-'))
                elif flow_range.startswith('>'):
                    low, high = float(flow_range[1:]), float('inf')
                else:
                    continue

                # unit: 万元
                cost = float(row['单位成本（万元/km）'])
                pipe.pipe_costs.append((low, high, cost))
            except (ValueError, TypeError) as e:
                print(f"Warning: Skipping invalid row: {row}, Error: {e}")
                continue

        return pipe
    
    def get_pipe_cost(self, flow):
        """
        Get the cost for a given flow.
        """
        for low, high, cost in self.pipe_costs:
            if low <= flow <= high:
                return cost
        
        # Return highest cost if flow exceeds all ranges
        return self.pipe_costs[-1][2] 


class FlowSolver:
    """
    Solves minimum cost flow problems on directed graphs.

    Attributes:
        graph (nx.DiGraph): The input graph with edge distances and node balances
        pipe (Pipe): Pipe object for cost calculations
        problem (LpProblem): The linear programming problem
        flow_vars (Dict[tuple, LpVariable]): Flow variables for each edge
    """        

    def __init__(self, G: nx.DiGraph, pipe: Pipe) -> None:  
        """
        Args:
            graph: Directed graph with edge distances and node balances
            pipe: Pipe object for cost calculations
        """      
        self.graph = G
        self.pipe = pipe
        self.problem = LpProblem("Hydrogen_Transport_Optimization", LpMinimize)
        self.flow_vars: Dict[tuple, LpVariable] = {}
        self._solution: Optional[Dict[str, Any]] = None

    def build_model(self) -> None:
        self._add_flow_vars()
        self._add_objective()
        self._add_flow_conservation_constraints()


    def _add_flow_vars(self) -> None:
        """
        Add flow variables to the problem.
        
        """
        for u, v, data in self.graph.edges(data=True):
            var = LpVariable(f"f_{u}_{v}", lowBound=0) # init flow = (lowBound, upBound) = (0, inf)
            self.flow_vars[(u, v)] = var


    def _add_objective(self) -> None:
        """
        Add objective function to the problem.
        Define Objective Function: Total Cost = Σ (flow on edge × distance of edge)
        """
        self.problem += lpSum(
            self.flow_vars[(u, v)] * data['distance']
            for u, v, data in self.graph.edges(data=True)
        )
    
    def _add_flow_conservation_constraints(self) -> None:
        """
        Add flow conservation constraints to the problem.
        """
        # u -> v
        for node, data in self.graph.nodes(data=True):
            inflow = lpSum(self.flow_vars[(u, v)] 
                           for (u, v) in self.flow_vars if v == node)
            outflow = lpSum(self.flow_vars[(u, v)] 
                            for (u, v) in self.flow_vars if u == node)
            self.problem += (inflow - outflow == data['balance'])

    def solve(self) -> Dict[str, Any]:
        """
        Solve the optimization problem.
        
        Returns:
            Dict containing:
                - status: Solution status
                - objective_value: Optimal objective value
                - flows: Dictionary of edge flows
        Raises:
            RuntimeError: If model hasn't been built or solver fails
        """
        if not self.flow_vars:
            raise RuntimeError("Model must be built before solving")
        
        # solve the LP problem
        status = self.problem.solve()
        
        if status != 1:
            self._solution = {
                'status': LpStatus[status],
                'objective_value': None,
                'flows': None
            }
            return self._solution
            
        self._solution = {
            'status': 'Optimal',
            'objective_value': self.problem.objective.value(),
            'flows': {
                edge: var.value() 
                for edge, var in self.flow_vars.items()
            }
        }
        return self._solution
    
    def get_result_df(self) -> pd.DataFrame:
        """
        Generate a DataFrame with the solution results.
        
        Returns:
            DataFrame containing flow results with columns:
            - 起点: Source node
            - 终点: Target node
            - 运输量（万吨）: Flow amount
            - 距离（km）: Edge distance
            - 单位成本（万元/km）: Cost per km (if pipe cost available)
            - 运输总成本（万元）: Total cost (if pipe cost available)
        """
        if not self._solution or self._solution['status'] != 'Optimal':
            raise RuntimeError("No optimal solution available")
        
        results: List[List] = []
        for (u, v), flow in self._solution['flows'].items():
            if flow and flow > 0:
                dist = self.graph[u][v]['distance']
                row = [u, v, flow, dist]

                if self.pipe:
                    cost_per_km = self.pipe.get_pipe_cost(flow)
                    total_cost = flow * dist * cost_per_km
                    row.extend([cost_per_km, total_cost])
                
                results.append(row)

        # Define columns
        columns = ["起点", "终点", "运输量（万吨）", "距离（km）"]
        if self.pipe:
            columns.extend(["单位成本（万元/km）", "运输总成本（万元）"])

        return pd.DataFrame(results, columns=columns)
    
    def save_result(self, filename: str) -> None:
        """
        Save the results to a CSV file.
        
        Args:
            filename: Name of the output file
        """
        df = self.get_results_dataframe()
        df.to_csv(filename, index=False)

