import cudaq
import networkx as nx
import numpy as np
from cudaq import spin


def get_K2_Hamiltonian(G: nx.Graph) -> cudaq.SpinOperator:
    """
    Generate Hamiltonian for k=2

    Args:
        G: Problem as a graph
        weights: Edge weights
        nodes: nodes of the graph
        add_identity: Add identiy or not. Defaults to False.

    Returns:
        Hamiltonian
    """
    H = 0

    for i, j in G.edges():
        weight = G[i][j]["weight"]
        H += weight * (spin.z(i) * spin.z(j))

    return H
