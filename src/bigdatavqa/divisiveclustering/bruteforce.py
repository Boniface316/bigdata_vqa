from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

from ..coreset import coreset_to_graph, normalize_np
from ..postexecution import add_children_to_hierachial_clustering


def perform_bruteforce_divisive_clustering(coreset_pd, method):
    coreset_pd = coreset_pd.rename(
        columns={"X_norm": "X", "Y_norm": "Y", "weights_norm": "weights"}
    )

    i = 0
    single_clusters = 0
    index_vals = [i for i in coreset_pd.index]

    while single_clusters < len(index_vals):
        if i < 1:
            hc = []
            hc.append(index_vals)

        if len(hc[i]) == 1:
            single_clusters += 1
            i += 1
        else:
            sub_index_vals = hc[i]
            sub_df = coreset_pd.iloc[sub_index_vals]

            if method == "random":
                bitstring_not_accepted = True
                while bitstring_not_accepted:
                    bitstring = create_cluster_using_random(sub_df)
                    if bitstring.sum() == 0 or bitstring.sum() == len(bitstring):
                        bitstring_not_accepted = True
                    else:
                        bitstring_not_accepted = False
            elif method == "kmeans":
                if len(sub_df) > 2:
                    bitstring = create_cluster_using_kmeans(sub_df)
                else:
                    bitstring = np.array([0, 1])

            elif method == "maxcut":
                qubits = len(sub_df)
                bitstring = get_best_bitstring(qubits, sub_df, sub_index_vals)
            else:
                raise ValueError("Method not found")

            hc = add_children_to_hierachial_clustering(sub_df, hc, bitstring)

            i += 1
    return hc


def create_cluster_using_random(df):
    rows = df.shape[0]
    return np.random.randint(0, 2, rows)


def create_cluster_using_kmeans(df):
    df = df.drop("name", axis=1)
    X = df.to_numpy()
    kmeans = KMeans(n_clusters=2, random_state=None).fit(X)
    return kmeans.labels_


def create_cluster_using_max_cut(df):
    pass


def get_best_bitstring(qubits: int, coreset_pd, idx_vals: int):
    """
    Finds the best bitstring out of all results

    Args:
        qubits: number of qubits for
        cw: coreset weights
        cv: coreset vectors
        idx_vals: index value at the hierarchy

    Returns:
        Best string value and cost value
    """

    cv = coreset_pd[["X", "Y"]].to_numpy()

    cw = coreset_pd["weights"].to_numpy()

    cv = normalize_np(cv, centralize=True)
    cw = normalize_np(cw, centralize=True)

    G, _ = coreset_to_graph(cv, cw)

    bitstrings = create_bitstrings(qubits)

    cost_val = brute_force_cost_2(bitstrings, G)

    cost_val_pd = create_cost_val_pd(cost_val)

    max_bitstring = get_max_bitstring(cost_val_pd)

    return max_bitstring


def create_bitstrings(qubits):
    """
    Using the number of qubits, it creates 2**qubits bitstrings

    Args:
        qubits: number of qubits

    Returns:
        All possible bitstrings
    """

    max_number = 2**qubits

    bit_length = len(format(max_number - 1, "b"))

    n_bits = "0" + str(bit_length) + "b"

    bitstrings = []
    for i in range(1, (2**qubits) - 1):
        bitstrings.append(format(i, n_bits))

    return bitstrings


def brute_force_cost_2(bitstrings: list, G: nx.graph):
    """
    Cost function for brute force method

    Args:
        bitstrings: list of bit strings
        G: The graph of the problem

    Returns:
       Dictionary with bitstring and cost value
    """
    cost_val = {}
    # use tdqm to show progress bar

    bitstrings_iter = tqdm(bitstrings)
    for bitstring in bitstrings_iter:
        c = 0
        for i, j in G.edges():
            ai = bitstring[i]
            aj = bitstring[j]
            ai = int(ai)
            aj = int(aj)

            weight_val = 1 * G[i][j]["weight"]
            c += cost_func_2(ai, aj, weight_val)

        cost_val.update({bitstring: c})

    return cost_val


def create_cost_val_pd(cost_val: Dict):
    """
    Converts the dictionary to a data frame

    Args:
        cost_val (Dict): Dictionary of cost

    Returns:
        data frame of all bitstring and cost
    """
    cost_val_pd = pd.DataFrame.from_dict(cost_val, orient="index")

    cost_val_pd.columns = ["cost"]

    cost_val_pd = cost_val_pd.sort_values("cost")

    cost_val_pd.reset_index()

    return cost_val_pd


def get_max_bitstring(cost_val_pd: pd.DataFrame):
    """
    Finds the bit string with high probability

    Args:
        cost_val_pd (pd.DataFrame): Dictionary with cost

    Returns:
        Bit string with high cost
    """
    max_cost_index = cost_val_pd[cost_val_pd["cost"] == cost_val_pd["cost"].max()]

    max_bitstrings = max_cost_index.index

    max_bitstring = max_bitstrings[0]

    max_bitstring_np = np.empty(1)

    for c in max_bitstring:
        max_bitstring_np = np.append(max_bitstring_np, int(c))

    max_bitstring = np.delete(max_bitstring_np, 0)

    return max_bitstring


def cost_func_2(a_i: int, a_j: int, weight_val: float):
    """Finds the cost value

    Args:
        a_i (int): Edge value 1
        a_j (int): Edge value 2
        weight_val (float): Edge weight

    Returns:
        _type_: _description_
    """

    val = -1 * weight_val * (1 - ((-1) ** a_i) * ((-1) ** a_j))  # MaxCut equation
    return val
