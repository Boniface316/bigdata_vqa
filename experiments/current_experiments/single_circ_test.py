import cudaq
import networkx as nx
import numpy as np
from cudaq import spin

qubits = 3
layers = 1


def kernel_two_local(number_of_qubits, circuit_depth) -> cudaq.Kernel:
    """QAOA ansatz for maxcut"""
    kernel, thetas = cudaq.make_kernel(list)
    qreg = kernel.qalloc(number_of_qubits)

    # Loop over the layers
    theta_position = 0

    for i in range(circuit_depth):
        for j in range(1, number_of_qubits):
            kernel.rz(thetas[theta_position], qreg[j % number_of_qubits])
            kernel.rx(thetas[theta_position + 1], qreg[j % number_of_qubits])
            kernel.cx(qreg[j], qreg[(j + 1) % number_of_qubits])
            kernel.rz(thetas[theta_position + 2], qreg[j % number_of_qubits])
            kernel.rx(thetas[theta_position + 3], qreg[j % number_of_qubits])
            theta_position += 4

    return kernel


kernel = kernel_two_local(qubits, layers)


def create_Hamiltonian_for_K2(
    G, qubits, weights: np.ndarray = None, add_identity=False
):
    H = 0

    for i, j in G.edges():
        weight = G[i][j]["weight"]  # [0]
        H += weight * (spin.z(i) * spin.z(j))

    return H


# create a fully connected graph with 3 nodes and equal weights
G = nx.complete_graph(3)
for i, j in G.edges():
    G[i][j]["weight"] = 1

# create a numpy array of weights for the edges
weights = np.array([G[i][j]["weight"] for i, j in G.edges()])


hamiltonian = create_Hamiltonian_for_K2(G, qubits, weights)


optimizer = cudaq.optimizers.COBYLA()
optimizer.initial_parameters = np.random.uniform(
    -np.pi / 8.0, np.pi / 8.0, 4 * layers * (qubits - 1)
)

max_shots = 1000
print(max_shots)

optimal_expectation, optimal_parameters = cudaq.vqe(
    # kernel=kernel_two_local(number_of_qubits, circuit_depth),
    kernel=kernel,
    spin_operator=hamiltonian,
    optimizer=optimizer,
    parameter_count=(qubits - 1) * 4 * layers,
    shots=max_shots,
)
