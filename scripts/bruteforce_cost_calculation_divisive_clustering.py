import pickle
import sys
import warnings

import numpy as np
import pandas as pd

from bigdatavqa.divisiveclustering.bruteforce import (
    perform_bruteforce_divisive_clustering,
)
from bigdatavqa.divisiveclustering.dendrogram import Dendrogram

warnings.filterwarnings("ignore")

coreset_pd = pd.read_csv("data/coreset.csv")

raw_data = pd.read_pickle("data/dataset.pickle")

method = sys.argv[1]

use_normalized = bool(sys.argv[2] == "True")


if method in ["random", "kmeans", "maxcut"]:
    if use_normalized:
        coreset_pd = coreset_pd[["X_norm", "Y_norm", "weights_norm", "name"]]
    else:
        coreset_pd = coreset_pd[["X", "Y", "weights", "name"]]
    hc = perform_bruteforce_divisive_clustering(coreset_pd, method)
else:
    use_normalized = True
    try:
        with open(f"data/hc_{method}_cut.pkl", "rb") as f:
            hc = pickle.load(f)
    except FileNotFoundError:
        print("File not found, exiting")
        exit()

    coreset_pd = coreset_pd[["X_norm", "Y_norm", "weights_norm", "name"]]


dendo = Dendrogram(
    hierarchial_clustering_sequence=hc,
    coreset_data=coreset_pd,
    raw_data=raw_data,
    use_normalized_coreset=use_normalized,
)


centroid_coords = dendo.get_centroid_coords()
cost_list = []
for parent_posn in range(len(hc)):
    children_lst = dendo.find_children(parent_posn)

    if len(children_lst) == 0:
        continue
    else:
        index_vals_temp = hc[parent_posn]
        child_1 = children_lst[0]
        child_2 = children_lst[1]

        if isinstance(index_vals_temp[0], str):
            index_vals_temp = [ord(c) - ord("A") for c in index_vals_temp]
            child_1 = [ord(c) - ord("A") for c in child_1]
            child_2 = [ord(c) - ord("A") for c in child_2]

        new_df = dendo.coreset_data.iloc[index_vals_temp]

        child_1_str = str(child_1)
        child_2_str = str(child_2)

        new_df["cluster"] = 0
        new_df.loc[child_1, "cluster"] = 1

        cost = 0

        for idx, row in new_df.iterrows():
            if row.cluster == 0:
                cost += (
                    np.linalg.norm(row[["X", "Y"]] - centroid_coords[child_1_str]) ** 2
                )
            else:
                cost += (
                    np.linalg.norm(row[["X", "Y"]] - centroid_coords[child_2_str]) ** 2
                )

        cost_list.append(cost)


print(f"{method}: {sum(cost_list)}")
