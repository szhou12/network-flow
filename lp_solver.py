from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import time
from datetime import datetime
from itertools import combinations
import networkx as nx
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus
import pulp as pl

class Edges:

    def __init__(self, file_path: str, from_colname: str, to_colname: str, dist_colname: str, is_dist_km: bool):
        self.edges_df = pd.read_csv(file_path).dropna()
        self.from_colname = from_colname
        self.to_colname = to_colname
        self.dist_colname = dist_colname
        self.is_dist_km = is_dist_km

    def convert_column_types(self, column_types: Dict[str, type]) -> None:
        """
        Convert specified columns to their requested types.
        
        Args:
            column_types: Dictionary mapping column names to their desired types
                         e.g. {'from_colname': int, 'to_colname': int}
        """
        for col, dtype in column_types.items():
            if col in self.edges_df.columns:
                self.edges_df[col] = self.edges_df[col].astype(dtype)
            else:
                print(f"Warning: Column '{col}' not found in DataFrame")

    
    def find_duplicates(self) -> pd.DataFrame:
        """
        Find all duplicate rows where from_colname and to_colname values are the same.
        
        Returns:
            DataFrame containing only the duplicate rows, sorted by from_colname and to_colname.
            If no duplicates are found, returns an empty DataFrame.
        """
        # Find duplicates based on from_colname and to_colname
        duplicates = self.edges_df[self.edges_df.duplicated(
            subset=[self.from_colname, self.to_colname], 
            keep=False
        )]
        
        # Sort by from_colname and to_colname for better readability
        if not duplicates.empty:
            duplicates = duplicates.sort_values([self.from_colname, self.to_colname])
            
        return duplicates
    
class Nodes:

    def __init__(self, file_path: str, id_colname: str, loc_colname: str, balance_colname: str):
        self.nodes_df = pd.read_csv(file_path).dropna()
        self.id_colname = id_colname
        self.loc_colname = loc_colname
        self.balance_colname = balance_colname
        self._normalize_balances()

    def convert_column_types(self, column_types: Dict[str, type]) -> None:
        """
        Convert specified columns to their requested types.
        
        Args:
            column_types: Dictionary mapping column names to their desired types
        """
        for col, dtype in column_types.items():
            if col in self.nodes_df.columns:
                self.nodes_df[col] = self.nodes_df[col].astype(dtype)
            else:
                print(f"Warning: Column '{col}' not found in DataFrame")

    def _normalize_balances(self, tolerance: float = 1e-10) -> None:
        """
        Normalize balances to ensure they sum to zero.
        If the sum is within tolerance of zero, adjust the largest absolute value to compensate.
        
        Args:
            tolerance: Maximum allowed deviation from zero sum
        """
        total_balance = self.nodes_df[self.balance_colname].sum()
        
        if abs(total_balance) > tolerance:
            print(f"total supply and demand is {total_balance} != 0, normalizing...")
            # Find the index of the largest absolute balance
            largest_idx = self.nodes_df[self.balance_colname].abs().idxmax()
            # Adjust the largest balance to make sum zero
            self.nodes_df.loc[largest_idx, self.balance_colname] -= total_balance
        
        print("normalizing done!")


class PipeCost:

    def __init__(self, file_path: str, flow_colname: str = '流量范围（万吨）', cost_colname: str = '单位成本（万元/km）'):
        self.pipe_cost_df = pd.read_csv(file_path).dropna()
        self.flow_colname = flow_colname
        self.cost_colname = cost_colname


def construct_graph(*, nodes: Nodes, edges: Edges = None):
    # directed graph
    G = nx.DiGraph()

    for _, row in nodes.nodes_df.iterrows():
        # negate demand value for the convenience of flow conservation constraint: 
        # (inflow - outflow) == balance
        # supply node needs to satisfy (inflow - outflow) = -supply (negative)
        # demand node needs to satisfy (inflow - outflow) = demand (positive)
        G.add_node(row[nodes.id_colname], balance = -row[nodes.balance_colname])
    
    if edges is None:
        # fully-connected bi-directional graph
        V = nodes.nodes_df[nodes.id_colname].unique().tolist()
        E = [(u, v) for u, v in combinations(V, 2)]
        for u, v in E:
            # default 200km
            G.add_edge(u, v, distance=200)
            G.add_edge(v, u, distance=200)
        return G
    
    # add directed edges in both directions instead of undirected edges
    # so that one flow variable corresponds to one directed edge
    # if edges_df is given, make sure each row is one-directional (u -> v). 
    # if u-v needs to be bi-directional, resolve it in input edges_df (add two rows: u->v and v->u). NOT HERE!
    for _, row in edges.edges_df.iterrows():
        u, v = row[edges.from_colname], row[edges.to_colname]
        if edges.is_dist_km:
            G.add_edge(u, v, distance=row[edges.dist_colname])
            # Check if reversed edge exists in edges_df before adding
            reversed_exists = edges.edges_df[
                (edges.edges_df[edges.from_colname] == v) & 
                (edges.edges_df[edges.to_colname] == u)
            ].shape[0] > 0
            if not reversed_exists:
                G.add_edge(v, u, distance=row[edges.dist_colname])
        else:
            # convert m to km
            
            G.add_edge(u, v, distance=row[edges.dist_colname] / 1000)
            # Check if reversed edge exists in edges_df before adding
            reversed_exists = edges.edges_df[
                (edges.edges_df[edges.from_colname] == v) & 
                (edges.edges_df[edges.to_colname] == u)
            ].shape[0] > 0
            if not reversed_exists:
                G.add_edge(v, u, distance=row[edges.dist_colname] / 1000)


    # Check connectivity of supply nodes
    supply_nodes = [node for node, data in G.nodes(data=True) if data['balance'] < 0]  # negative balance = supply
    isolated = [node for node in supply_nodes 
               if not any(G.has_edge(node, v) or G.has_edge(v, node) 
                         for v in G.nodes())]
    
    if isolated:
        raise ValueError(f"Supply nodes {isolated} are disconnected")
    
    return G


# def construct_graph(*, nodes_df, edges_df=None, distance=200):
#     # directed graph
#     G = nx.DiGraph()

#     for _, row in nodes_df.iterrows():
#         # negate demand value for the convenience of flow conservation constraint: 
#         # (inflow - outflow) == balance
#         # supply node needs to satisfy (inflow - outflow) = -supply (negative)
#         # demand node needs to satisfy (inflow - outflow) = demand (positive)
#         G.add_node(row['省份编号'], balance = -row['供需量（万吨）'])

#     # add directed edges in both directions instead of undirected edges
#     # so that one flow variable corresponds to one directed edge
#     # if edges_df is given, make sure each row is one-directional (u -> v). if u-v needs to be bi-directional, resolve it in input edges_df (add two rows: u->v and v->u). NOT HERE!
#     if edges_df is not None:
#         valid_edges = edges_df[edges_df['是否可连 (1/0)'] == 1]
#         for _, row in valid_edges.iterrows():
#             u, v, dist = row['起点省份编号'], row['终点省份编号'], row['距离 (km)']
#             G.add_edge(u, v, distance=dist)
#         #     G.add_edge(v, u, distance=dist)
#         # G.add_edge(21, 20, distance=40)
#         # G.add_edge(20, 21, distance=40)
#     else:
#         # fully-connected bi-directional graph
#         n = nodes_df.shape[0]
#         V = range(1, n + 1)
#         E = [(u, v) for u, v in combinations(V, 2)]
#         for u, v in E:
#             G.add_edge(u, v, distance=distance)
#             G.add_edge(v, u, distance=distance)
        
#     return G


class Pipe:
    def __init__(self):
        self.pipe_costs = []
    
    @classmethod
    def from_data(
        cls, 
        pipe_cost: PipeCost
    ):
        """
        Create a Pipe object from input dataframe.

        """
        # validate dataframe containing required columns
        required_columns = [pipe_cost.flow_colname, pipe_cost.cost_colname]
        if not all(col in pipe_cost.pipe_cost_df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        # Init Pipe object
        pipe = cls()

        # load data
        for _, row in pipe_cost.pipe_cost_df.iterrows():
            try:
                flow_range = row[pipe_cost.flow_colname]
                if '-' in flow_range:
                    low, high = map(float, flow_range.split('-'))
                elif flow_range.startswith('>'):
                    low, high = float(flow_range[1:]), float('inf')
                else:
                    continue

                # unit: 万元
                cost = float(row[pipe_cost.cost_colname])
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
        
        # Enable solver logging
        pl.LpSolverDefault.msg = True
        
        # Configure CBC solver with debugging options
        cbc = pl.PULP_CBC_CMD(
            msg=True, 
            options=[
                "presolve on",   # run presolve
                "findiis",               # generate IIS if problem still infeasible
                "writeiis iis.ilp"       # save it to disk
            ]
        )
        
        # Solve with debug options
        status = self.problem.solve(cbc)
        
        # Write model to file for debugging
        self.problem.writeLP("debug.lp")
        
        if status != 1:
            print(f"Solver status: {LpStatus[status]}")
            print("Check debug.lp and iis.ilp files for more information")
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
                row = [int(u), int(v), flow, dist]  # Convert node IDs to integers

                if self.pipe:
                    cost_per_km = self.pipe.get_pipe_cost(flow)
                    total_cost = flow * dist * cost_per_km
                    row.extend([cost_per_km, total_cost])
                
                results.append(row)

        # Define columns
        columns = ["起点", "终点", "运输量（万吨）", "距离（km）"]
        if self.pipe:
            columns.extend(["单位成本（万元/km）", "运输总成本（万元）"])

        df = pd.DataFrame(results, columns=columns)
        # Ensure node ID columns are integers
        df["起点"] = df["起点"].astype(int)
        df["终点"] = df["终点"].astype(int)
        return df
    
    def save_result(self, filename: str) -> None:
        """
        Save the results to a CSV file.
        
        Args:
            filename: Name of the output file
        """
        df = self.get_results_dataframe()
        df.to_csv(filename, index=False)

