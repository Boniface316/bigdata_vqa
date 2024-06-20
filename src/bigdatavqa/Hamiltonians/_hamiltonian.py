import cudaq
import networkx as nx
from cudaq import spin


def get_K2_Hamiltonian(G: nx.Graph) -> cudaq.SpinOperator:
    """
    Get the Hamiltonian for the K2 model.

    Args:
        G (nx.Graph): The graph for which the Hamiltonian is to be calculated.

    Returns:
        cudaq.SpinOperator: The Hamiltonian for the K2 model.

    """

    H = 0

    for i, j in G.edges():
        weight = G[i][j]["weight"]
        H += weight * (spin.z(i) * spin.z(j))

    return H


def get_K3_Hamiltonian(G: nx.Graph) -> cudaq.SpinOperator:
    """

    Get the Hamiltonian for the K3 model.

    Args:
        G (nx.Graph): The graph for which the Hamiltonian is to be calculated.

    Returns:
        cudaq.SpinOperator: The Hamiltonian for the K3 model.

    """
    H = 0
    for i, j in G.edges():
        weight = G[i][j]["weight"]
        H += weight * (
            (5 * spin.i(i) * spin.i(i + 1) * spin.i(j) * spin.i(j + 1))
            + spin.z(i + 1)
            + spin.z(j + 1)
            - (spin.z(i) * spin.z(j))
            - (3 * spin.z(i + 1) * spin.z(j + 1))
            - (spin.z(i) * spin.z(i + 1) * spin.z(j))
            - (spin.z(i) * spin.z(j) * spin.z(j + 1))
            - (spin.z(i) * spin.z(i + 1) * spin.z(j) * spin.z(j + 1))
        )

    return -(1 / 8) * H


def get_GMM_Hamiltonian(pauli_operators) -> cudaq.SpinOperator:
    """
    Get the Hamiltonian for the GMM model.

    Args:
        pauli_operators (List[Tuple[str, float]]): The list of Pauli operators.

    Returns:
        cudaq.SpinOperator: The Hamiltonian for the GMM model.

    """

    H = 0
    for idx, op in enumerate(pauli_operators):
        operator_string = op[0]
        coeff = op[1]
        operator = 1

        for i in range(len(operator_string)):
            op_i = operator_string[i]
            if op_i == "Z":
                operator *= spin.z(i)
            if op_i == "I":
                operator *= spin.i(i)
        H += coeff * operator

    return -1 * H
