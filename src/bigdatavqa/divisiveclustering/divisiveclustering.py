import cudaq
from loguru import logger

from ..coreset import (
    Coreset,
    get_coreset_vector_df,
    get_coreset_vectors_to_evaluate,
    coreset_to_graph,
    get_cv_cw,
)
from ..optimizer import get_optimizer
from ..postexecution import add_children_to_hierachial_clustering, get_best_bitstring
from ..vqe_utils import (
    kernel_two_local,
)
import numpy as np
from cudaq import spin
import cudaq
import pandas as pd
from typing import Optional


def get_coreset_vec_and_weights(
    raw_data,
    number_of_qubits,
    number_of_centroids_evaluation,
    number_of_coresets_to_evaluate,
):
    coreset = Coreset()
    return coreset.get_best_coresets(
        data_vectors=raw_data,
        number_of_runs=number_of_centroids_evaluation,
        coreset_numbers=number_of_qubits,
        size_vec_list=number_of_coresets_to_evaluate,
    )


def create_hierarchial_cluster(
    raw_data,
    number_of_qubits,
    number_of_centroids_evaluation,
    number_of_coresets_to_evaluate,
    max_shots,
    max_iterations,
    circuit_depth,
):
    coreset_vectors, coreset_weights = get_coreset_vec_and_weights(
        raw_data,
        number_of_qubits,
        number_of_centroids_evaluation,
        number_of_coresets_to_evaluate,
    )
    breakpoint()
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

            optimal_expectation, optimal_parameters = cudaq.vqe(
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


def get_hierarchial_clustering_sequence(coreset_weights):
    index_values = [i for i in range(len(coreset_weights))]
    hioerarchial_clustering_sequence = [index_values]

    return index_values, hioerarchial_clustering_sequence


def get_hamiltonian_coreset_vector_G(
    coreset_vectors,
    coreset_weights,
    coreset_vector_df,
    hierarchial_clustering_sequence,
    index_iteration_counter,
    add_identity=False,
):
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
        coreset_weights, coreset_vectors = get_cv_cw(
            coreset_vectors, coreset_weights, index_vals_temp
        )

    G, weights = coreset_to_graph(coreset_vectors, coreset_weights, metric="dot")
    qubits = len(G.nodes)

    return G, weights, qubits


def create_Hamiltonian_for_K2(
    G, qubits, weights: np.ndarray = None, add_identity=False
):
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
        weight = G[i][j]["weight"]  # [0]
        H += weight * (spin.z(i) * spin.z(j))

    return H
