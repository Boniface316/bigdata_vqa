import argparse
import warnings
from pathlib import Path

import cudaq
from loguru import logger

from bigdatavqa.coreset import (
    Coreset,
    get_coreset_vector_df,
    get_coreset_vectors_to_evaluate,
)
from bigdatavqa.datautils import DataUtils
from bigdatavqa.optimizer import get_optimizer
from bigdatavqa.postexecution import (
    add_children_to_hierachial_clustering,
    get_best_bitstring,
)
from bigdatavqa.vqe_utils import (
    create_Hamiltonian_for_K2,
    get_Hamiltonian_variables,
    kernel_two_local,
)

parser = argparse.ArgumentParser(description="Divisive clustering circuit parameters")

parser.add_argument("--qubits", type=int, required=True, help="Number of qubits")
parser.add_argument("--layers", type=int, required=True, help="Number of layers")
parser.add_argument("--shots", type=int, required=True, help="Number of shots")
parser.add_argument(
    "--iterations", type=int, required=True, help="Number of iterations"
)
parser.add_argument("--data_location", type=str, required=False, help="Data location")
args = parser.parse_args()


logger.add(
    ".logs/divisive_clustering.log",
    rotation="10 MB",
    compression="zip",
    level="INFO",
    retention="10 days",
)


number_of_qubits = args.qubits
circuit_depth = args.layers
max_shots = args.shots
max_iterations = args.iterations
if args.data_location is None:
    data_location = "data"
else:
    data_location = args.data_location

max_iterations = 100
number_of_runs = 100
size_vec_list = 10

logger.info(f"Number of qubits: {number_of_qubits}")
logger.info(f"Number of layers: {circuit_depth}")
logger.info(f"Number of shots: {max_shots}")
logger.info(f"Number of iterations: {max_iterations}")
logger.info(f"Data location: {data_location}")


def get_raw_data(data_location):
    data_utils = DataUtils(data_location)

    try:
        raw_data = data_utils.load_dataset()
    except FileNotFoundError:
        raw_data = data_utils.create_dataset(n_samples=1000)

    return raw_data


def get_coreset_vectors(raw_data, number_of_runs, coreset_numbers, size_vec_list):
    coreset = Coreset()

    return coreset.get_best_coresets(
        data_vectors=raw_data,
        number_of_runs=number_of_runs,
        coreset_numbers=coreset_numbers,
        size_vec_list=size_vec_list,
    )


def main(
    data_location,
    number_of_qubits,
    circuit_depth,
    max_shots,
    max_iterations,
    number_of_runs,
    size_vec_list,
):
    index_iteration_counter = 0
    single_clusters = 0

    raw_data = get_raw_data(data_location)
    coreset_vectors, coreset_weights = get_coreset_vectors(
        raw_data, number_of_runs, number_of_qubits, size_vec_list
    )

    coreset_vector_df = get_coreset_vector_df(coreset_vectors, index_iteration_counter)

    index_values = [i for i in range(len(coreset_weights))]
    hierarchial_clustering_sequence = [index_values]

    while single_clusters < len(index_values):
        if len(hierarchial_clustering_sequence[index_iteration_counter]) == 1:
            single_clusters += 1
            logger.info(
                f"index iteration counter : {index_iteration_counter},  Single clusters count: {single_clusters}"
            )
        else:
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
            logger.info(
                f"index iteration counter : {index_iteration_counter}, QUBITS :{qubits}, index values to evaluate: {index_values_to_evaluate}",
            )

            Hamiltonian = create_Hamiltonian_for_K2(
                G, qubits, weights, add_identity=False
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
            coreset_vectors_df_to_evaluate["cluster"] = [int(bit) for bit in bitstring]
            hierarchial_clustering_sequence = add_children_to_hierachial_clustering(
                coreset_vectors_df_to_evaluate, hierarchial_clustering_sequence
            )
            logger.info(
                f"index iteration counter: {index_iteration_counter} Last splits: {hierarchial_clustering_sequence[-1]} {hierarchial_clustering_sequence[-2]} "
            )

        index_iteration_counter += 1
    logger.info(
        f"Final results from hierarchial clustering: {hierarchial_clustering_sequence}"
    )


if __name__ == "__main__":
    raw_data = get_raw_data(data_location)
    coreset = Coreset()
    coreset_vectors, coreset_weights = coreset.get_best_coresets(
        data_vectors=raw_data,
        number_of_runs=number_of_runs,
        coreset_numbers=number_of_qubits,
        size_vec_list=size_vec_list,
    )

    main(
        data_location,
        number_of_qubits,
        circuit_depth,
        max_shots,
        max_iterations,
        number_of_runs,
        size_vec_list,
    )
