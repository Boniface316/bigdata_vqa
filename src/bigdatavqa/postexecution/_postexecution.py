from typing import List

import networkx as nx
import numpy as np
import pandas as pd


def get_divisive_cluster_cost(dendo, hc, centroid_coords):
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

            child_1_str_list = [str_val for str_val in child_1]

            new_df["cluster"] = 0
            new_df.loc[new_df.name.isin(child_1_str_list), "cluster"] = 1

            cost = 0

            for idx, row in new_df.iterrows():
                if row.cluster == 0:
                    cost += (
                        np.linalg.norm(row[["X", "Y"]] - centroid_coords[child_1]) ** 2
                    )
                else:
                    cost += (
                        np.linalg.norm(row[["X", "Y"]] - centroid_coords[child_2]) ** 2
                    )

            cost_list.append(cost)

    return cost_list


def get_distance_between_two_vectors(
    vector1: np.ndarray, vector2: np.ndarray, weight: np.ndarray
) -> float:
    return weight * np.linalg.norm(vector1 - vector2)


def get_k_means_accumulative_cost(k, clusters, data, data_weights=None):
    accumulativeCost = 0
    currentCosts = np.repeat(0, k)
    data_weights = np.repeat(1, len(data)) if data_weights is None else data_weights
    for vector in data:
        currentCosts = list(
            map(
                get_distance_between_two_vectors,
                clusters,
                np.repeat(vector, k, axis=0),
                data_weights,
            )
        )
        accumulativeCost = accumulativeCost + min(currentCosts)

    return accumulativeCost
