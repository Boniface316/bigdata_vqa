from typing import List, Optional, Tuple

import cudaq
import networkx as nx
import numpy as np
import pandas as pd
from cudaq import spin
from loguru import logger

from ..coreset import (
    Coreset,
    coreset_to_graph,
    get_coreset_vector_df,
    get_coreset_vectors_to_evaluate,
    get_cv_cw,
)
from ..optimizer import get_optimizer
from ..postexecution import add_children_to_hierachial_clustering, get_best_bitstring
from ..vqe_utils import kernel_two_local


def get_coreset_vec_and_weights(
    raw_data: np.ndarray,
    number_of_qubits: int,
    number_of_centroids_evaluation: int,
    number_of_coresets_to_evaluate: int,
):
    """
    Get the coreset vectors and weights for divisive clustering.

    Args:
        raw_data (np.ndarray): The raw data vectors.
        number_of_qubits (int): The number of qubits for coreset compression.
        number_of_centroids_evaluation (int): The number of centroids to evaluate.
        number_of_coresets_to_evaluate (int): The number of coreset vectors to evaluate.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The coreset vectors and weights.
    """
    coreset = Coreset()
    return coreset.get_best_coresets(
        data_vectors=raw_data,
        number_of_runs=number_of_centroids_evaluation,
        coreset_size=number_of_qubits,
        size_vec_list=number_of_coresets_to_evaluate,
    )


def create_hierarchial_cluster(
    raw_data: np.ndarray,
    number_of_qubits: int,
    number_of_centroids_evaluation: int,
    number_of_coresets_to_evaluate: int,
    max_shots: int = 1000,
    max_iterations: int = 1000,
    circuit_depth: int = 1,
) -> Tuple[List[int], List[np.ndarray]]:
    """
    Creates a hierarchical cluster using divisive clustering algorithm.

    Args:
        raw_data (np.ndarray): The input data for clustering.
        number_of_qubits (int): The number of qubits to be used in the clustering.
        number_of_centroids_evaluation (int): The number of centroids to evaluate when creating the coreset.
        number_of_coresets_to_evaluate (int): The number of coreset vectors to evaluate when creating the coreset.
        max_shots (int, optional): The maximum number of shots for quantum measurements. Defaults to 1000.
        max_iterations (int, optional): The maximum number of iterations for the clustering algorithm. Defaults to 1000.
        circuit_depth (int, optional): The depth of the quantum circuit. Defaults to 1.

    Returns:
        Tuple[List[int], List[np.ndarray]]: A tuple containing the hierarchical clustering sequence and the coreset vectors and weights.

    """

    coreset_vectors, coreset_weights = get_coreset_vec_and_weights(
        raw_data,
        number_of_qubits,
        number_of_centroids_evaluation,
        number_of_coresets_to_evaluate,
    )

    index_iteration_counter = 0
    single_clusters = 0

    coreset_vector_df = get_coreset_vector_df(coreset_vectors, index_iteration_counter)

    index_values, hierarchial_clustering_sequence = get_hierarchial_clustering_sequence(
        coreset_weights
    )

    while single_clusters < len(index_values):
        if len(hierarchial_clustering_sequence[index_iteration_counter]) == 1:
            single_clusters += 1
            logger.info(
                f"index iteration counter : {index_iteration_counter},  Single clusters count: {single_clusters}"
            )
        else:
            (
                Hamiltonian,
                coreset_vectors_df_to_evaluate,
                G,
            ) = get_hamiltonian_coreset_vector_G(
                coreset_vectors,
                coreset_weights,
                coreset_vector_df,
                hierarchial_clustering_sequence,
                index_iteration_counter,
            )

            qubits = len(G.nodes)

            logger.info(
                f"index iteration counter : {index_iteration_counter}, QUBITS :{qubits}",
            )

            optimizer, parameter_count = get_optimizer(
                max_iterations, circuit_depth, qubits
            )

            _, optimal_parameters = cudaq.vqe(
                kernel=kernel_two_local(qubits, circuit_depth),
                spin_operator=Hamiltonian[0],
                optimizer=optimizer,
                parameter_count=parameter_count,
                shots=max_shots,
            )

            counts = cudaq.sample(
                kernel_two_local(qubits, circuit_depth),
                optimal_parameters,
                shots_count=max_shots,
            )

            bitstring = get_best_bitstring(counts, qubits, G)
            logger.info(
                f"index iteration counter : {index_iteration_counter}, Bitstring: {bitstring}"
            )

            hierarchial_clustering_sequence = add_children_to_hierachial_clustering(
                coreset_vectors_df_to_evaluate,
                hierarchial_clustering_sequence,
                bitstring,
            )
            logger.info(
                f"index iteration counter: {index_iteration_counter} Last splits: {hierarchial_clustering_sequence[-1]} {hierarchial_clustering_sequence[-2]} "
            )
        index_iteration_counter += 1
    logger.success(
        f"Final results from hierarchial clustering: {hierarchial_clustering_sequence}"
    )

    return hierarchial_clustering_sequence, [coreset_vectors, coreset_weights]


def get_hierarchial_clustering_sequence(
    coreset_weights: np.ndarray,
) -> Tuple[List, List]:
    """
    Returns the index values and the hierarchical clustering sequence for the start

    Args:
        coreset_weights (np.ndarray): The weights of the coreset.

    Returns:
        Tuple[List, List]: A tuple containing the index values and the hierarchical clustering sequence.
    """
    index_values = [i for i in range(len(coreset_weights))]
    hierarchial_clustering_sequence = [index_values]

    return index_values, hierarchial_clustering_sequence


def get_hamiltonian_coreset_vector_G(
    coreset_vectors: np.ndarray,
    coreset_weights: np.ndarray,
    coreset_vector_df: pd.DataFrame,
    hierarchial_clustering_sequence: List,
    index_iteration_counter,
    add_identity=False,
) -> Tuple[cudaq.SpinOperator, pd.DataFrame, nx.Graph]:
    """
    Calculates the Hamiltonian, coreset vector for the evaluation and the fully connected weighted graph.

    Args:
        coreset_vectors (np.ndarray): The coreset vectors.
        coreset_weights (np.ndarray): The weights of the coreset vectors.
        coreset_vector_df (pd.DataFrame): The DataFrame containing the coreset vectors.
        hierarchial_clustering_sequence (List): The hierarchical clustering sequence.
        index_iteration_counter: The index iteration counter.
        add_identity (bool, optional): Whether to add identity. Defaults to False.

    Returns:
        Tuple[cudaq.SpinOperator, pd.DataFrame, nx.Graph]: The Hamiltonian coreset vector G, the DataFrame of coreset vectors to evaluate, and the graph G.
    """

    (
        coreset_vectors_df_to_evaluate,
        index_values_to_evaluate,
    ) = get_coreset_vectors_to_evaluate(
        coreset_vector_df,
        hierarchial_clustering_sequence,
        index_iteration_counter,
    )

    G, weights, qubits = get_Hamiltonian_variables(
        coreset_vectors,
        coreset_weights,
        index_values_to_evaluate,
        coreset_vectors_df_to_evaluate,
    )

    return (
        create_Hamiltonian_for_K2(G, qubits, weights, add_identity=add_identity),
        coreset_vectors_df_to_evaluate,
        G,
    )


def get_Hamiltonian_variables(
    coreset_vectors: np.ndarray,
    coreset_weights: np.ndarray,
    index_vals_temp: Optional[int] = None,
    new_df: Optional[pd.DataFrame] = None,
) -> Tuple[nx.Graph, np.ndarray, int]:
    """
    Generates the variables required to create the Hamiltonian

    Args:
        coreset_vectors: Coreset vectors
        coreset_weights: Coreset weights
        index_vals_temp: Index in the hierarchy
        new_df: new dataframe create for this problem,

    Returns:
       Graph, weights and qubits
    """
    if new_df is not None and index_vals_temp is not None:
        coreset_weights, coreset_vectors = get_cv_cw(
            coreset_vectors, coreset_weights, index_vals_temp
        )

    G, weights = coreset_to_graph(coreset_vectors, coreset_weights, metric="dot")
    qubits = len(G.nodes)

    return G, weights, qubits


def create_Hamiltonian_for_K2(
    G: nx.Graph, qubits: int, weights: np.ndarray = None, add_identity=False
) -> cudaq.SpinOperator:
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
