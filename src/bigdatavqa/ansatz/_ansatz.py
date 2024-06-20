import cudaq


def get_VQE_circuit(number_of_qubits: int, circuit_depth: int) -> cudaq.Kernel:
    """

    Get the VQE circuit.

    Args:
        number_of_qubits (int): The number of qubits in the circuit.
        circuit_depth (int): The depth of the circuit.

    Returns:
        cudaq.Kernel: The VQE circuit


    """

    @cudaq.kernel
    def kernel(thetas: list[float], number_of_qubits: int, circuit_depth: int):
        qubits = cudaq.qvector(number_of_qubits)

        theta_position = 0

        for i in range(circuit_depth):
            for j in range(number_of_qubits):
                ry(thetas[theta_position], qubits[j])
                rz(thetas[theta_position + 1], qubits[j])
                theta_position += 2

            for j in range(number_of_qubits - 1):
                cx(qubits[j], qubits[j + 1])

            for j in range(number_of_qubits):
                ry(thetas[theta_position], qubits[j])
                rz(thetas[theta_position + 1], qubits[j])

                theta_position += 2

    return kernel


def get_QAOA_circuit(number_of_qubits: int, circuit_depth: int) -> cudaq.Kernel:
    """

    Get the QAOA circuit.

    Args:
        number_of_qubits (int): The number of qubits in the circuit.
        circuit_depth (int): The depth of the circuit.

    Returns:
        cudaq.Kernel: The QAOA circuit

    """

    @cudaq.kernel
    def kernel(thetas: list[float], number_of_qubits: int, circuit_depth: int):
        qubits = cudaq.qvector(number_of_qubits)

        layers = circuit_depth

        for layer in range(layers):
            for qubit in range(number_of_qubits):
                cx(qubits[qubit], qubits[(qubit + 1) % number_of_qubits])
                rz(2.0 * thetas[layer], qubits[(qubit + 1) % number_of_qubits])
                cx(qubits[qubit], qubits[(qubit + 1) % number_of_qubits])

            rx(2.0 * thetas[layer + layers], qubits)

    return kernel
