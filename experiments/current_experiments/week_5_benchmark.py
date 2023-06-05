from divisiveclustering.datautils import DataUtils
from divisiveclustering.coresetsUtils import Coreset, normalize_np, gen_coreset_graph
from divisiveclustering.vqe_utils import (
    kernel_two_local,
    create_Hamiltonian_for_K2,
    get_Hamil_variables,
)
from divisiveclustering.quantumutils import get_probs_table
import cudaq
import numpy as np
import warnings
import datetime
import sys

cudaq.set_qpu("custatevec")

warnings.filterwarnings("ignore")

# time_dict = {}
# simulator_options = ["custatevec", "custatevec_f32"]


number_of_qubits = int(sys.argv[1])
layer_count = 1
parameter_count = 4 * layer_count * number_of_qubits
shots = int(sys.argv[2])


data_utils = DataUtils()

raw_data = data_utils.create_dataset(n_samples=1000, save_file=False)

coresets = Coreset()

coreset_vectors, coreset_weights = coresets.get_coresets(
    data_vectors=raw_data,
    number_of_runs=10,
    coreset_numbers=number_of_qubits,
    size_vec_list=10,
)

best_coreset_vectors, best_coreset_weights = coresets.get_best_coresets(
    raw_data, coreset_vectors, coreset_weights
)

normalized_cv = normalize_np(best_coreset_vectors, centralize=True)
normalized_cw = normalize_np(best_coreset_weights, centralize=False)

coreset_points, G, H, weight_matrix, weights = gen_coreset_graph(
    normalized_cv, normalized_cw, metric="dot"
)

H = create_Hamiltonian_for_K2(G, number_of_qubits, weights)

optimizer = cudaq.optimizers.COBYLA()
optimizer.initial_parameters = np.random.uniform(
    -np.pi / 8.0, np.pi / 8.0, parameter_count
)
print(optimizer.initial_parameters)

# breakpoint()

print(datetime.datetime.now())
start_time = datetime.datetime.now()
optimal_expectation, optimal_parameters = cudaq.vqe(
    kernel=kernel_two_local(number_of_qubits, layer_count),
    spin_operator=H,
    optimizer=optimizer,
    parameter_count=parameter_count,
    shots=shots,
)

counts = cudaq.sample(
    kernel_two_local(number_of_qubits, layer_count),
    optimal_parameters,
    shots_count=shots,
)

counts.dump()

print(f"end time:{datetime.datetime.now()}")
print(f"total time:{datetime.datetime.now() - start_time}")
