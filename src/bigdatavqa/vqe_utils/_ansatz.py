import cudaq


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


def create_Hamiltonian_for_K2(
    G: nx.Graph, qubits: int, weights: np.ndarray = None, add_identity=False
) -> cudaq.SpinOperator:
    """
    Generate Hamiltonian for k=2

    Args:
        G: Problem as a graph
        weights: Edge weights
        nodes: nodes of the graph
        add_identity: Add identiy or not. Defaults to False.

    Returns:
        Hamiltonian
    """
    H = 0

    for i, j in G.edges():
        weight = G[i][j]["weight"]
        H += weight * (spin.z(i) * spin.z(j))

    return H[0]
