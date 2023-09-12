import datetime
import pickle
import re

import networkx as nx
import pandas as pd
from divisiveclustering.coresetsUtils import coreset_to_graph
from divisiveclustering.datautils import DataUtils
from divisiveclustering.quantumutils import QAOA_divisive_clustering
from sklearn.cluster import KMeans

coreset_numbers = 5
qubits = coreset_numbers
centers = 4
depth = 1

data_util = DataUtils()
cv, cw, data_vec = data_util.get_files(coreset_numbers, centers)

coreset_points, G, H, weight_matrix, weights = coreset_to_graph(cv, cw, metric="dot")

df = pd.DataFrame(cv, columns=list("XY"))

df["Name"] = [chr(i + 65) for i in df.index]

hc = QAOA_divisive_clustering(
    df,
    cw,
    cv,
    coreset_numbers=coreset_numbers,
    centers=centers,
    depth=depth,
    shots=100,
    max_iterations=1000,
    step_size=0.01,
)
