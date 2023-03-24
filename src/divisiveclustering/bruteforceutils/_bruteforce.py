from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from divisiveclustering.coresetsUtils import gen_coreset_graph, get_cv_cw


def create_clusters(
    type: str,
    df: pd.DataFrame,
    qubits: int = None,
    cw: np.ndarray = None,
    cv: np.ndarray = None,
    idx_vals: int = None,
):
    """_summary_

    Args:
        type: Type of algorithm to use to perform clustering
        df: dataframe of the data set that we want to cluster
        qubits: Number of qubits needed for this algorithm. Defaults to None.
        cw : Coreset weights. Defaults to None.
        cv : Coreset vectors. Defaults to None.
        idx_vals: Index value in the hierarchy. Defaults to None.

    Returns:
        clusters and cost value of the iterations
    """

    cost_val_pd = None
    if type == "random":
        rows = df.shape[0]
        clusters = np.random.randint(0, 2, rows)
    elif type == "kmeans":
        X = df.to_numpy()
        kmeans = KMeans(n_clusters=2, random_state=None).fit(X)
        clusters = kmeans.labels_
    elif type == "maxcut":
        clusters, cost_val_pd = get_best_bitstring(qubits, cw, cv, idx_vals)

    return clusters, cost_val_pd


def get_best_bitstring(qubits: int, cw: np.ndarray, cv: np.ndarray, idx_vals: int):

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

    cw, cv = get_cv_cw(cv, cw, idx_vals)

    coreset_points, G, H, weight_matrix, weights = gen_coreset_graph(
        cv, cw, metric="dot"
    )

    bitstrings = create_bitstrings(qubits)

    cost_val = brute_force_cost_2(bitstrings, G)

    cost_val_pd = create_cost_val_pd(cost_val)

    max_bitstring = get_max_bitstring(cost_val_pd)

    return max_bitstring, cost_val_pd


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
    for bitstring in bitstrings:

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
