import numpy as np
import pandas as pd
from divisiveclustering.bruteforceutils import brute_force_cost_2
from divisiveclustering.coresetsUtils import coreset_to_graph
from divisiveclustering.datautils import DataUtils
from divisiveclustering.helpers import find_children, np_to_bitstring
from divisiveclustering.quantumutils import get_Hamil_variables

type = "VQE"
coresets = 5
centers = 4
depth = 1

data_utils = DataUtils()

coreset_vectors, coreset_weights, data_vec = data_utils.get_files(coresets, centers)

coreset_points, G, H, weight_matrix, weights = coreset_to_graph(
    coreset_weights, coreset_vectors, metric="dot"
)

df = pd.DataFrame(coreset_vectors, columns=list("XY"))

hc = data_utils.load_object(type, coresets, centers, depth, "hc")

cost_list = []

for parent_posn in range(len(hc)):
    children_lst = find_children(hc, parent_posn)

    if len(children_lst) == 0:
        continue
    else:
        index_vals_temp = hc[parent_posn]
        new_df = df.iloc[index_vals_temp]

        new_df["cluster"] = 0
        new_df.loc[children_lst[0], "cluster"] = 1

        str_np = np.array(new_df.cluster)

        bitstrings = [np_to_bitstring(str_np)]

        G, weights, qubits = get_Hamil_variables(
            coreset_vectors, coreset_weights, index_vals_temp, new_df
        )

        cost_val_dict = brute_force_cost_2(bitstrings, G)

        cost_list.append(cost_val_dict[bitstrings[0]][0])


print(sum(cost_list))
