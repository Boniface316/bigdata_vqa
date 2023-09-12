import datetime
import os
import pickle
import sys
import warnings
from typing import Dict, List, Optional

import cudaq
import networkx as nx
import numpy as np
import pandas as pd
from cudaq import spin
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans


def kernel_two_local(number_of_qubits, circuit_depth) -> cudaq.Kernel:
    """QAOA ansatz for maxcut"""
    kernel, thetas = cudaq.make_kernel(list)
    qreg = kernel.qalloc(number_of_qubits)

    # Loop over the layers
    theta_position = 0

    for i in range(circuit_depth):
        for j in range(number_of_qubits):
            kernel.rz(thetas[theta_position], qreg[j % number_of_qubits])
            kernel.rx(thetas[theta_position + 1], qreg[j % number_of_qubits])
            kernel.cx(qreg[j], qreg[(j + 1) % number_of_qubits])
            kernel.rz(thetas[theta_position + 2], qreg[j % number_of_qubits])
            kernel.rx(thetas[theta_position + 3], qreg[j % number_of_qubits])
            theta_position += 4

    return kernel


def create_Hamiltonian_for_K2(qubits):
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
    G = nx.complete_graph(qubits)

    for i, j in G.edges():
        weight = np.random.uniform(0, 1)
        H += weight * (spin.z(i) * spin.z(j))

    return H[0]


warnings.filterwarnings("ignore")

time_dict = {}
simulator_options = [None, "custatevec", "custatevec_f32", "cuquantum_mgpu"]
# simulator_options = ["cuquantum"]

number_of_qubits = int(sys.argv[1])
circuit_depth = 1
parameter_count = 4 * circuit_depth * number_of_qubits
shots = int(sys.argv[2])

H = create_Hamiltonian_for_K2(number_of_qubits)

optimizer = cudaq.optimizers.COBYLA()
optimizer.initial_parameters = np.random.uniform(
    -np.pi / 8.0, np.pi / 8.0, parameter_count
)


# breakpoint()
for simulator in simulator_options:
    if simulator is not None:
        cudaq.set_qpu(simulator)
    print(datetime.datetime.now())
    start_time = datetime.datetime.now()
    optimal_expectation, optimal_parameters = cudaq.vqe(
        kernel=kernel_two_local(number_of_qubits, circuit_depth),
        spin_operator=H,
        optimizer=optimizer,
        parameter_count=parameter_count,
        shots=shots,
    )

    print(f"end time:{datetime.datetime.now()}")
    print(f"total time:{datetime.datetime.now() - start_time}")
    if simulator is None:
        simulator = "cpu"
    time_dict[simulator] = datetime.datetime.now() - start_time


print(time_dict)
