import cudaq
from cudaq import spin
import networkx as nx
import numpy as np
from typing import Optional
import pandas as pd
from divisiveclustering.coresetsUtils import get_cv_cw, gen_coreset_graph

def kernel_two_local(number_of_qubits, layer_count) -> cudaq.Kernel:
    """QAOA ansatz for maxcut"""
    kernel, thetas = cudaq.make_kernel(list)
    qreg = kernel.qalloc(number_of_qubits)

    # Loop over the layers
    theta_position = 0
    
    for i in range(layer_count):

        for j in range(number_of_qubits):
            kernel.rz(thetas[theta_position], qreg[j % number_of_qubits])
            kernel.rx(thetas[theta_position + 1], qreg[j % number_of_qubits])
            kernel.cx(qreg[j], qreg[(j + 1) % number_of_qubits])
            kernel.rz(thetas[theta_position + 2], qreg[j % number_of_qubits])
            kernel.rx(thetas[theta_position + 3], qreg[j % number_of_qubits])
            theta_position += 4

    return kernel


def get_Hamil_variables(
    coreset_vectors: np.ndarray,
    coreset_weights: np.ndarray,
    index_vals_temp: Optional[int] = None,
    new_df: Optional[pd.DataFrame] = None,
):
    """
    Generates the variables required for Hamiltonian

    Args:
        coreset_vectors: Coreset vectors
        coreset_weights: Coreset weights
        index_vals_temp: Index in the hierarchy
        new_df: new dataframe create for this problem,

    Returns:
       Graph, weights and qubits
    """
    if new_df is not None and index_vals_temp is not None:
        coreset_weights, coreset_vectors = get_cv_cw(coreset_vectors, coreset_weights, index_vals_temp)
    
    coreset_points, G, H, weight_matrix, weights = gen_coreset_graph(
        coreset_vectors, coreset_weights, metric="dot"
    )
    qubits = len(G.nodes)

    return G, weights, qubits

def create_Hamiltonian_for_K2(G, qubits, weights: np.ndarray = None,add_identity=False):
    """
    Generate Hamiltonian for k=2

    Args:
        G: Problem as a graph
        weights: Edge weights
        nodes: nodes of the graph
        add_identity: Add identiy or not. Defaults to False.

    Returns:
        _type_: _description_
    """
    H = 0

    for i, j in G.edges():
        weight = G[i][j]["weight"]#[0]
        H += weight * (spin.z(i) * spin.z(j))
        
    return H[0]