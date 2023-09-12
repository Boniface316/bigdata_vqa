import warnings

import cudaq
import numpy as np
import pandas as pd
from divisiveclustering.coresetsUtils import Coreset, coreset_to_graph
from divisiveclustering.datautils import DataUtils
from divisiveclustering.helpers import add_children_to_hc
from divisiveclustering.quantumutils import get_probs_table
from divisiveclustering.vqe_utils import (
    create_Hamiltonian_for_K2,
    get_Hamil_variables,
    kernel_two_local,
)

number_of_qubits = 20
circuit_depth = 1

data_utils = DataUtils("data")

try:
    raw_data = data_utils.load_dataset()
except:
    raw_data = data_utils.create_dataset(n_samples=1000)

coresets = Coreset()

coreset_vectors, coreset_weights = coresets.get_coresets(
    data_vectors=raw_data,
    number_of_runs=100,
    coreset_numbers=number_of_qubits,
    size_vec_list=10,
)

coreset_vectors, coreset_weights = coresets.get_best_coresets(
    raw_data, coreset_vectors, coreset_weights
)

coreset_vectors = np.array(coreset_vectors)
coreset_weights = np.array(coreset_weights)

coreset_points, G, H, weight_matrix, weights = coreset_to_graph(
    coreset_vectors, coreset_weights, metric="dot"
)

df = pd.DataFrame(coreset_vectors, columns=list("XY"))

df["Name"] = [chr(i + 65) for i in df.index]

index_vals = [i for i in range(len(coreset_weights))]
hc = [index_vals]

i = 0
single_clusters = 0
max_iterations = 1000
max_shots = 100000

error_counter = 0

while single_clusters < len(index_vals):
    print(f"Iteration: {i}")
    if i > 0:
        hc = data_utils.load_object("VQE", "hc")
    if len(hc[i]) == 1:
        single_clusters += 1
        i += 1
    else:
        index_values_to_evaluate = hc[i]
        df_to_evaluate = df.iloc[index_values_to_evaluate]
        df_to_evaluate = df_to_evaluate.drop(columns=["Name"])
        G, weights, qubits = get_Hamil_variables(
            coreset_vectors, coreset_weights, index_values_to_evaluate, df_to_evaluate
        )
        print(f"Qubits: {qubits}")
        hamiltonian = create_Hamiltonian_for_K2(G, number_of_qubits, weights)

        parameter_count = 4 * circuit_depth * (qubits - 1)

        optimizer = cudaq.optimizers.COBYLA()
        optimizer.initial_parameters = np.random.uniform(
            -np.pi / 8.0, np.pi / 8.0, parameter_count
        )
        optimizer.max_iterations = max_iterations

        optimal_expectation, optimal_parameters = cudaq.vqe(
            kernel=kernel_two_local(qubits, circuit_depth),
            spin_operator=hamiltonian[0],
            optimizer=optimizer,
            parameter_count=parameter_count,
            shots=max_shots,
        )

        counts = cudaq.sample(
            kernel_two_local(qubits, circuit_depth),
            optimal_parameters,
            shots_count=max_shots,
        )
        probs_table = get_probs_table(
            counts, qubits, sort_values=True, remove_zero_one=True
        )
        try:
            best_bitstring = probs_table.iloc[0][0]
        except:
            # pick a number between 1 and 2^qubits-1. Convert that to binary and pad with 0s at the front to match the qubits length.
            best_bitstring = np.binary_repr(
                np.random.randint(1, 2**qubits - 1), width=qubits
            )
            error_counter += 1

        # TODO: remove the 000 and 111
        print(probs_table)
        print(best_bitstring)
        print(qubits)

        df_to_evaluate["clusters"] = [int(i) for i in best_bitstring]
        hc = add_children_to_hc(df_to_evaluate, hc)

        data_utils.save_object("VQE", hc)
        # save object

        i += 1

print(hc)
print(f"Error counter: {error_counter}")
