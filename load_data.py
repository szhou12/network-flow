import numpy as np
import pandas as pd
import time
from datetime import datetime

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