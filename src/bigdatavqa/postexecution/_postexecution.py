from typing import List

import networkx as nx
import numpy as np
import pandas as pd


def get_probs_table(counts, qubits):
    all_bitstrings = _get_bitstrings(qubits)

    return _crearte_bitstring_probs_df(all_bitstrings, counts)


def _get_bitstrings(qubits: int) -> List[str]:
    all_bitstrings = []
    for i in range(1, (2**qubits - 1)):
        all_bitstrings.append(bin(i)[2:].zfill(qubits))
    return all_bitstrings


def _crearte_bitstring_probs_df(all_bitstrings, counts, sort_values: bool = True):
    df = pd.DataFrame(columns=["bitstring", "probability"])
    for bitstring in all_bitstrings:
        df.loc[len(df)] = [bitstring, counts.probability(bitstring)]

    if sort_values:
        df = df.sort_values("probability", ascending=False)

    return df


def get_best_bitstring(counts, G, threshold=0.5):
    bitstring_probability_df = get_probs_table(counts, len(G.nodes))
    # get top percentage of the bitstring_probability_df
    if len(bitstring_probability_df) > 100:
        selected_rows = int(len(bitstring_probability_df) * threshold)
    else:
        selected_rows = int(len(bitstring_probability_df) / 2)
    bitstring_probability_df = bitstring_probability_df.head(selected_rows)

    bitstrings = bitstring_probability_df["bitstring"].tolist()

    brute_force_cost_of_bitstrings = brute_force_cost_maxcut(bitstrings, G)

    return min(brute_force_cost_of_bitstrings, key=brute_force_cost_of_bitstrings.get)


def add_children_to_hierachial_clustering(df: pd.DataFrame, hc: list, bitstring: str):
    """
    Add the children to the hierachy structure

    Args:
        df: dataframe
        hc: hierachy structure

    Returns:
        updated list with children
    """
    df["cluster"] = [int(bit) for bit in bitstring]

    for j in range(2):
        idx = list(df[df["cluster"] == j].index)
        if len(idx) > 0:
            hc.append(idx)

    return hc


def brute_force_cost_maxcut(bitstrings: list, G: nx.graph):
    """
    Cost function for brute force method

    Args:
        bitstrings: list of bit strings
        G: The graph of the problem

    Returns:
       Dictionary with bitstring and cost value
    """
    cost_value = {}
    for bitstring in bitstrings:
        c = 0
        for i, j in G.edges():
            c += bitstring_cost_using_maxcut(bitstring, i, j, G[i][j]["weight"])

        cost_value.update({bitstring: c})

    return cost_value


def bitstring_cost_using_maxcut(bitstring: str, i, j, edge_weight):
    """Finds the cost value

    Args:
        a_i (int): Edge value 1
        a_j (int): Edge value 2
        weight_val (float): Edge weight

    Returns:
        _type_: _description_
    """
    ai = int(bitstring[i])
    aj = int(bitstring[j])

    edge_weight = edge_weight

    val = -1 * edge_weight * (1 - ((-1) ** ai) * ((-1) ** aj))  # MaxCut equation
    return val


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
