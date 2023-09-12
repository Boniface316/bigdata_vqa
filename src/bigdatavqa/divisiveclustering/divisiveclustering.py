import cudaq
from loguru import logger

from ..coreset import get_coreset_vector_df, get_coreset_vectors_to_evaluate
from ..optimizer import get_optimizer
from ..postexecution import add_children_to_hierachial_clustering, get_best_bitstring
from ..vqe_utils import (
    create_Hamiltonian_for_K2,
    get_Hamiltonian_variables,
    kernel_two_local,
)


def create_hierarchial_cluster(
    coreset_vectors, coreset_weights, max_shots, max_iterations, circuit_depth
):
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
    logger.info(
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
