import cudaq
import networkx as nx
import numpy as np
from cudaq import spin


def get_K2_Hamiltonian(G: nx.Graph) -> cudaq.SpinOperator:
    H = 0

    for i, j in G.edges():
        weight = G[i][j]["weight"]
        H += weight * (spin.z(i) * spin.z(j))

    return H


def get_K3_Hamiltonian(G: nx.Graph) -> cudaq.SpinOperator:
    H = 0
    for i, j in G.edges():
        weight = G[i][j]["weight"]
        H += weight * (
            (5 * spin.i(i) * spin.i(i + 1) * spin.i(j) * spin.i(j + 1))
            + spin.z(i + 1)
            + spin.z(j + 1)
            - (spin.z(i) * spin.z(j))
            - (3 * spin.z(i + 1) * spin.z(j + 1))
            - (spin.z(i) * spin.z(i + 1) * spin.z(j))
            - (spin.z(i) * spin.z(j) * spin.z(j + 1))
            - (spin.z(i) * spin.z(i + 1) * spin.z(j) * spin.z(j + 1))
        )

    return -(1 / 8) * H
