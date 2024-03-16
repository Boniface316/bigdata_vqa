from typing import List, Optional, Tuple

import cudaq
import networkx as nx
import numpy as np
import pandas as pd
from cudaq import spin
from loguru import logger

from ..coreset import Coreset, coreset_to_graph
from ..optimizer import get_optimizer
from ..postexecution import add_children_to_hierachial_clustering, get_best_bitstring
from ..vqe_utils import create_Hamiltonian_for_K2, kernel_two_local


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
        number_of_coresets_to_evaluate=number_of_coresets_to_evaluate,
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

    initial_coreset_vectors, initial_coreset_weights = get_coreset_vec_and_weights(
        raw_data,
        number_of_qubits,
        number_of_centroids_evaluation,
        number_of_coresets_to_evaluate,
    )

    index_iteration_counter = 0
    single_clusters = 0

    index_values = [i for i in range(len(initial_coreset_vectors))]
    hierarchial_clustering_sequence = [index_values]

    full_coreset_vectors_df = pd.DataFrame(initial_coreset_vectors, columns=list("XY"))

    full_coreset_vectors_df["Name"] = [
        chr(i + 65) for i in full_coreset_vectors_df.index
    ]

    while single_clusters < len(index_values):
        if len(hierarchial_clustering_sequence[index_iteration_counter]) == 1:
            single_clusters += 1
            logger.info(
                f"index iteration counter : {index_iteration_counter},  Single clusters count: {single_clusters}"
            )
        else:
            index_values_to_evaluate = hierarchial_clustering_sequence[
                index_iteration_counter
            ]
            coreset_vectors_df_for_iteration = initial_coreset_vectors.iloc[
                index_values_to_evaluate
            ]

            coreset_vectors_for_iteration, coreset_weights_for_iteration = (
                get_iteration_coreset_vectors_and_weights(
                    index_values_to_evaluate,
                    initial_coreset_vectors,
                    initial_coreset_weights,
                )
            )

            G = coreset_to_graph(
                coreset_vectors_for_iteration,
                coreset_weights_for_iteration,
                metric="dot",
            )

            Hamiltonian = create_Hamiltonian_for_K2(G)

            qubits = len(G.nodes)

            kernel = kernel_two_local(qubits, circuit_depth)

            logger.info(
                f"index iteration counter : {index_iteration_counter}, QUBITS :{qubits}",
            )

            optimizer, parameter_count = get_optimizer(
                max_iterations, circuit_depth, qubits
            )

            _, optimal_parameters = cudaq.vqe(
                kernel=kernel,
                spin_operator=Hamiltonian,
                optimizer=optimizer,
                parameter_count=parameter_count,
                shots=max_shots,
            )

            counts = cudaq.sample(
                kernel,
                optimal_parameters,
                shots_count=max_shots,
            )

            bitstring = get_best_bitstring(counts, qubits, G)
            logger.info(
                f"index iteration counter : {index_iteration_counter}, Bitstring: {bitstring}"
            )

            hierarchial_clustering_sequence = add_children_to_hierachial_clustering(
                coreset_vectors_df_for_iteration,
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


def get_iteration_coreset_vectors_and_weights(
    index_values_to_evaluate,
    initial_coreset_vectors,
    initial_coreset_weights,
    normalize=True,
):

    coreset_weights_for_iteration = initial_coreset_weights[index_values_to_evaluate]
    coreset_vectors_for_iteration = initial_coreset_vectors[index_values_to_evaluate]

    if normalize:

        coreset_vectors_for_iteration = normalize_array(
            coreset_vectors_for_iteration, True
        )
        coreset_weights_for_iteration = normalize_array(coreset_weights_for_iteration)

    return coreset_vectors_for_iteration, coreset_weights_for_iteration
