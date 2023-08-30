import warnings

import cudaq
import numpy as np
import pandas as pd

from bigdatavqa.coreset import Coreset, gen_coreset_graph, get_coreset_vector_df, get_coreset_vectors_to_evaluate
from bigdatavqa.datautils import DataUtils
from bigdatavqa.vqe_utils import (
    create_Hamiltonian_for_K2,
    get_Hamiltoniam_variables,
    kernel_two_local,
)

from bigdatavqa.postexecution import get_probs_table

from bigdatavqa.optimizer import get_optimizer

from loguru import logger

logger.add(".logs/recreate_divisive_clustering.log", rotation="10 MB", compression="zip", level="INFO")

number_of_qubits = 7
layer_count = 1

index_iteration_counter = 0
single_clusters = 0
max_iterations = 1000
max_shots = 100000
error_counter = 0


data_utils = DataUtils("data")

try:
    raw_data = data_utils.load_dataset()
except:
    raw_data = data_utils.create_dataset(n_samples=1000)

coreset = Coreset()

coreset_vectors, coreset_weights = coreset.get_best_coresets(data_vectors = raw_data, number_of_runs = 100, coreset_numbers = number_of_qubits, size_vec_list = 10)

#coreset_points, G, H, weight_matrix, weights = gen_coreset_graph(coreset_vectors, coreset_weights, metric = "dot")

coreset_vector_df = get_coreset_vector_df(coreset_vectors, index_iteration_counter)


index_values = [i for i in range(len(coreset_weights))]
hierarchial_clustering_sequence = [index_values]


while single_clusters < len(index_values):
    # use log to save the iteration number
    logger.info(f"Current index iteration:{index_iteration_counter}")

    if len(hc[i]) == 1:
        single_clusters += 1
        index_iteration_counter += 1
    else:
        coreset_vectors_df_to_evaluate, index_values_to_evaluate = get_coreset_vectors_to_evaluate(coreset_vector_df, hierarchial_clustering_sequence, index_iteration_counter)
        G, weights, qubits = get_Hamiltoniam_variables(coreset_vectors, coreset_weights, index_values_to_evaluate, coreset_vectors_df_to_evaluate)
        logger.info(f"Number of qubits: {qubits}")
        Hamiltonian = create_Hamiltonian_for_K2(G, qubits, weights, add_identity = False)
        logger.info(f"Hamiltonian: {Hamiltonian}")

        optimizer, parameter_count = get_optimizer(max_iterations, layer_count, qubits)

        optimal_expectation, optimal_parameters = cudaq.vqe(
            kernel=kernel_two_local(qubits, layer_count),
            spin_operator=Hamiltonian[0],
            optimizer=optimizer,
            parameter_count=parameter_count,
            shots=max_shots,
        )

        counts = cudaq.sample(
            kernel_two_local(qubits, layer_count),
            optimal_parameters,
            shots_count=max_shots,
        )
        probs_table = get_probs_table(
            counts, qubits, sort_values=True, remove_zero_one=True
        )

        breakpoint()



