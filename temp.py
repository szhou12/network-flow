
from datetime import datetime
import pandas as pd
import networkx as nx
from pulp import LpProblem, LpMinimize, LpVariable, lpSum

input_dir = "data/"
output_dir = "results/"

# 读取数据
supply_demand_df = pd.read_csv(input_dir + "mismatch by province.csv")
pipeline_cost_df = pd.read_csv(input_dir + "pipeline cost.csv")
connection_df = pd.read_csv(input_dir + "province connection.csv")

# 构建图
G = nx.DiGraph()
for _, row in supply_demand_df.iterrows():
    G.add_node(row['省份编号'], demand=-row['供需量（万吨）'])

valid_edges = connection_df[connection_df['是否可连 (1/0)'] == 1]
for _, row in valid_edges.iterrows():
    u, v, dist = row['起点省份编号'], row['终点省份编号'], row['距离 (km)']
    G.add_edge(u, v, distance=dist)
    G.add_edge(v, u, distance=dist)

# 添加海南 ↔ 广东连接
G.add_edge(21, 20, distance=40)
G.add_edge(20, 21, distance=40)

# 管径成本函数
pipe_costs = []
for _, row in pipeline_cost_df.iterrows():
    fr = row['流量范围（万吨）']
    if '-' in fr:
        low, high = map(float, fr.split('-'))
    elif fr.startswith('>'):
        low, high = float(fr[1:]), float('inf')
    else:
        continue
    cost = float(row['单位成本（万元/km）']) * 10000
    pipe_costs.append((low, high, cost))

print(pipe_costs)

def get_pipe_cost(flow):
    for low, high, cost in pipe_costs:
        if low <= flow <= high:
            return cost
    return pipe_costs[-1][2]

# [(0.0, 0.5, 220000.0), 
#  (0.5, 2.9, 830000.0), 
#  (2.9, 6.4, 1880000.0), 
#  (6.4, 10.0, 2870000.0), 
#  (10.0, 39.9, 11470000.0), 
#  (39.9, inf, 20000000.0)]

# 定义线性规划问题
prob = LpProblem("Hydrogen_Transport_Optimization", LpMinimize)
flow_vars = {}
for u, v, data in G.edges(data=True):
    var = LpVariable(f"f_{u}_{v}", lowBound=0) # init flow = (lowBound, upBound) = (0, inf)
    flow_vars[(u, v)] = var

# Objective fcn: Total Cost = Σ (flow on edge × distance of edge × cost per unit distance)
# prob += lpSum(flow_vars[(u, v)] * data['distance'] * get_pipe_cost(0.001) for u, v, data in G.edges(data=True))
# prob += lpSum(flow_vars[(u, v)] * data['distance'] * get_pipe_cost(60) for u, v, data in G.edges(data=True))
prob += lpSum(flow_vars[(u, v)] * data['distance'] for u, v, data in G.edges(data=True))

# 添加流量守恒约束
for node, data in G.nodes(data=True):
    inflow = lpSum(flow_vars[(u, v)] for (u, v) in flow_vars if v == node)
    outflow = lpSum(flow_vars[(u, v)] for (u, v) in flow_vars if u == node)
    prob += (inflow - outflow == data['demand'])

# 求解
prob.solve()

# 输出结果
results = []
for (u, v), var in flow_vars.items():
    flow = var.varValue
    if flow and flow > 0:
        dist = G[u][v]['distance']
        cost_per_km = get_pipe_cost(flow)
        total_cost = flow * dist * cost_per_km
        results.append([u, v, flow, dist, cost_per_km, total_cost])

results_df = pd.DataFrame(results, columns=["起点", "终点", "运输量（万吨）", "距离（km）", "单位成本（元/km）", "运输总成本（元）"])

csv_filename = f'hydrogen_transport_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
results_df.to_csv(output_dir + csv_filename, index=False)
print(f'优化完成，结果已保存为 {csv_filename}')