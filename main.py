import numpy as np
import pandas as pd
import time
from datetime import datetime

from ortools_solver import ortools_solver
from network_design_solver import solve_design_network
from lp_solver import FlowSolver, Pipe, construct_graph

input_dir = "data/"
output_dir = "results/"

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
    supply_demand_df = pd.read_csv(input_dir + "mismatch by province.csv")
    connection_df = pd.read_csv(input_dir + "province connection.csv")
    pipeline_cost_df = pd.read_csv(input_dir + "pipeline cost.csv")

    # 构建图
    G = construct_graph(
        nodes_df=supply_demand_df,
        # edges_df=connection_df
    )

    # pipe cost
    pipe = Pipe.from_dataframe(pipeline_cost_df)

    print("building and solving...")

    # 求解
    solver = FlowSolver(G, pipe)
    solver.build_model()
    result = solver.solve()

    if result['status'] == 'Optimal':
        result_df = solver.get_result_df()
        csv_filename = f'network_flow_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        result_df.to_csv("results/"+csv_filename, index=False)
    else:
        print(result['status'])



if __name__ == "__main__":
    main()