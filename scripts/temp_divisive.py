import cudaq
import numpy as np
import pandas as pd
from cudaq import spin
from matplotlib import pyplot as plt

from bigdatavqa.coreset import Coreset, coreset_to_graph, normalize_np
from bigdatavqa.datautils import DataUtils
from bigdatavqa.divisiveclustering import Dendrogram, create_hierarchial_cluster
from bigdatavqa.divisiveclustering.bruteforce import (
    perform_bruteforce_divisive_clustering,
)
from bigdatavqa.postexecution import get_divisive_cluster_cost

qubits = 5
circuit_depth = 1
max_shots = 100
max_iterations = 10
number_of_experiment_runs = 5

number_of_corsets_to_evaluate = 15
number_of_centroid_evaluation = 20

data_utils = DataUtils("data")
raw_data = data_utils.load_dataset()

parameter_count = 4 * circuit_depth * (qubits - 1)

optimizer = cudaq.optimizers.COBYLA()
optimizer.initial_parameters = np.random.uniform(
    -np.pi / 8.0, np.pi / 8.0, parameter_count
)
optimizer.max_iterations = max_iterations

coreset = Coreset()
coreset_vectors, coreset_weights = coreset.get_best_coresets(
    data_vectors=raw_data,
    number_of_runs=number_of_centroid_evaluation,
    coreset_size=qubits,
    number_of_coresets_to_evaluate=number_of_corsets_to_evaluate,
)

G, weights = coreset_to_graph(coreset_vectors, coreset_weights, metric="dot")

H = 0

for i, j in G.edges():
    weight = G[i][j]["weight"]
    H += weight * (spin.z(i) * spin.z(j))

kernel, thetas = cudaq.make_kernel(list)
qreg = kernel.qalloc(qubits)

# Loop over the layers
theta_position = 0

for i in range(circuit_depth):
    for j in range(1, qubits):
        kernel.rz(thetas[theta_position], qreg[j % qubits])
        kernel.rx(thetas[theta_position + 1], qreg[j % qubits])
        kernel.cx(qreg[j], qreg[(j + 1) % qubits])
        kernel.rz(thetas[theta_position + 2], qreg[j % qubits])
        kernel.rx(thetas[theta_position + 3], qreg[j % qubits])
        theta_position += 4

optimal_parameters = cudaq.vqe(
    kernel=kernel,
    spin_operator=H,
    optimizer=optimizer,
    parameter_count=parameter_count,
    shots=max_shots,
)
