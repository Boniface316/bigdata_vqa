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
import networkx as nx
from cudaq import spin



warnings.filterwarnings("ignore")

time_dict = {}
simulator_options = [None, "custatevec", "custatevec_f32", "cuquantum_mgpu"]
#simulator_options = ["cuquantum"]

number_of_qubits = int(sys.argv[1])
layer_count = 1
parameter_count = 4 * layer_count * number_of_qubits
shots = int(sys.argv[2])


def create_Hamiltonian_for_K2(number_of_qubits):
    G = nx.complete_graph(number_of_qubits)
    H = 0

    for edge in G.edges():
        weight = np.random.uniform(0.0, 1.0)
        H += weight * (spin.z(edge[0]) * spin.z(edge[1]))
    
    return H[0]



H = create_Hamiltonian_for_K2(number_of_qubits)

optimizer = cudaq.optimizers.COBYLA()
optimizer.initial_parameters = np.random.uniform(
    -np.pi / 8.0, np.pi / 8.0, parameter_count
)
print(optimizer.initial_parameters)

# breakpoint()
for simulator in simulator_options:
    if simulator is not None:
        cudaq.set_qpu(simulator)
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
    if simulator is None:
        simulator = "cpu"
    time_dict[simulator] = datetime.datetime.now() - start_time


print(time_dict)