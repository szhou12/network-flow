import numpy as np
import pandas as pd
import time
from datetime import datetime

from ortools_solver import ortools_solver
from network_design_solver import solve_design_network
from lp_solver import FlowSolver, Pipe, construct_graph, Nodes, Edges, PipeCost

input_dir = "data/"
output_dir = "results/"

# deprecated
def laod_data():
    supply_demand_df = pd.read_csv("data/mismatch by province.csv")
    balance = supply_demand_df['供需量（万吨）'].tolist()

    packages = [
        [1, 0.5, 22],
        [2, 2.9, 83],
        [3, 6.4, 188],
        [4, 10, 287],
        [5, 39.9, 1147],
    ]


    connection_df = pd.read_csv("data/province connection.csv")
    valid_edges = connection_df[connection_df['是否可连 (1/0)'] == 1]
    connections = [(start - 1, end - 1) for start, end in valid_edges[['起点省份编号', '终点省份编号']].values]

    return balance, packages, connections


# def main():
#     # edges = [
#     #     [0, 2,  12, 2],   # A -> X
#     #     [0, 3,  12, 4],   # A -> Y
#     #     [0, 1,  12, 1],
#     #     [1, 2,  12, 3],   # B -> X
#     #     [1, 3,  12, 1],   # B -> Y
#     # ]

#     # balance = [ 7, 5, -4, -8 ]

#     # ortools_solver(edges, balance)

#     balance, packages, connections = laod_data()
#     dist = 200
    
#     # Record start time
#     start_time = time.time()
#     print("running...")
    
#     status, opt_cost, chosen, flow_sol = solve_design_network(
#         balances=balance,
#         packages=packages,
#         distance=dist,
#         connections=connections
#     )

#     print("done!")
    
#     # Calculate running time
#     running_time = time.time() - start_time
    
#     # Create results dictionary
#     results = {
#         'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#         'status': status,
#         'optimal_cost': opt_cost,
#         'chosen_packages': chosen,
#         'flow_solution': flow_sol,
#         'running_time_seconds': running_time
#     }
    
#     # Convert to DataFrame and save to CSV
#     df = pd.DataFrame([results])
#     csv_filename = f'network_flow_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
#     df.to_csv("results/"+csv_filename, index=False)
    
#     print(f"Results saved to {csv_filename}")
#     print(f"Running time: {running_time:.2f} seconds")
#     print(status, opt_cost, chosen, flow_sol)


def main():

    nodes_id_colname = 'fid'
    nodes_loc_colname = '地名'
    nodes_balance_colname = '供需差异'

    edges_from_colname = '起点ID'
    edges_to_colname = '终点ID'
    edges_dist_colname = 'length_m'

    pipe_cost_flow_colname = '流量范围（万吨）'
    pipe_cost_cost_colname = '单位成本（万元/km）'

    # nodes data file is ok
    nodes = Nodes(
        file_path=input_dir + "0529区域供需差异结果_输入py.csv", 
        id_colname=nodes_id_colname, 
        loc_colname=nodes_loc_colname,
        balance_colname=nodes_balance_colname
    )

    nodes.convert_column_types({
        nodes_id_colname: int,
        # nodes_loc_colname: str,
        # nodes_balance_colname: float
    })

    edges = Edges(
        file_path=input_dir + "筛选_连线及距离_1000.csv", 
        from_colname=edges_from_colname, 
        to_colname=edges_to_colname, 
        dist_colname=edges_dist_colname, 
        is_dist_km=False
    )

    edges.convert_column_types({
        edges_from_colname: int,
        edges_to_colname: int,
        # edges_dist_colname: int
    })



    pipe_cost = PipeCost(
        file_path=input_dir + "pipeline cost.csv", 
        flow_colname=pipe_cost_flow_colname, 
        cost_colname=pipe_cost_cost_colname
    )

    # check if there are duplicate edges
    dups = edges.find_duplicates()
    if not dups.empty:
        dup_filename = f'duplicate_edges_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        dups.to_csv(output_dir + dup_filename, index=False)
        print(f"Found duplicate edges. Saved to {dup_filename}")

    # 构建图
    try:
        G = construct_graph(
            nodes=nodes,
            edges=edges
        )
    except ValueError as e:
        print(f"Error: {e}")
        print("Please check the input data and connections.")
        exit(1)

    # pipe cost
    pipe = Pipe.from_data(pipe_cost)

    print("building and solving...")

    # 求解
    solver = FlowSolver(G, pipe)
    solver.build_model()
    result = solver.solve()

    # geo-spatial coordinates
    coord_mapping = pd.read_csv(input_dir + "区域供需差异及坐标.csv", usecols=['fid', 'X坐标', 'Y坐标'])

    if result['status'] == 'Optimal':
        result_df = solver.get_result_df(coord_mapping)
        csv_filename = f'network_flow_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        result_df.to_csv("results/"+csv_filename, index=False)
    else:
        print(result['status'])



if __name__ == "__main__":
    main()