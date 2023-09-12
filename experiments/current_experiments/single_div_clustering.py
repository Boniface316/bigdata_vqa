import warnings

import cudaq

from bigdatavqa.coreset import (
    Coreset,
    get_coreset_vector_df,
    get_coreset_vectors_to_evaluate,
)
from bigdatavqa.datautils import DataUtils
from bigdatavqa.optimizer import get_optimizer
from bigdatavqa.postexecution import get_probs_table
from bigdatavqa.vqe_utils import (
    create_Hamiltonian_for_K2,
    get_Hamiltonian_variables,
    kernel_two_local,
)

number_of_qubits = 4
circuit_depth = 1

index_iteration_counter = 0
single_clusters = 0
max_iterations = 100
max_shots = 100
error_counter = 0


data_utils = DataUtils("data")

try:
    raw_data = data_utils.load_dataset()
except FileNotFoundError:
    raw_data = data_utils.create_dataset(n_samples=1000)

coreset = Coreset()

coreset_vectors, coreset_weights = coreset.get_best_coresets(
    data_vectors=raw_data,
    number_of_runs=100,
    coreset_numbers=number_of_qubits,
    size_vec_list=10,
)

coreset_vector_df = get_coreset_vector_df(coreset_vectors, index_iteration_counter)

index_values = [i for i in range(len(coreset_weights))]
hierarchial_clustering_sequence = [index_values]

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

Hamiltonian = create_Hamiltonian_for_K2(G, qubits, weights, add_identity=False)

optimizer, parameter_count = get_optimizer(max_iterations, circuit_depth, qubits)

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
