from ..vqe_utils import kernel_two_local
from ..optimizer import get_optimizer
import cudaq
from ..coreset import Coreset, normalize_np
import numpy as np
from numpy.linalg import inv
from cudaq import spin


def get_mean(coreset_vectors):
    dim = len(coreset_vectors[0])
    size_coreset = len(coreset_vectors)
    mu = np.zeros(dim)
    for i in range(size_coreset):
        mu += coreset_vectors[i]
    return mu / size_coreset


def get_scatter_matrix(coreset_vectors):
    dim = len(coreset_vectors[0])
    size_coreset = len(coreset_vectors)
    T = np.zeros((dim, dim))
    mu = get_mean(coreset_vectors)
    for i in range(size_coreset):
        T += np.outer((coreset_vectors[i] - mu), (coreset_vectors[i] - mu))
    return T


def Z_i(i, length):
    """
    if index i is in the range 0, ..., length-1, the function returns the operator Z_i
    else: the funtion returns the pauli string consisting of pauli I's only
    length is the number of pauli operators tensorised
    """
    pauli_string = ""
    for j in range(length):
        if i == j:
            pauli_string += "Z"
        else:
            pauli_string += "I"
    return pauli_string


def Z_ij(i, j, length):
    pauli_string = ""
    if i == j:
        pauli_string = Z_i(-1, length)  # return 'II...II'
    else:
        for k in range(length):
            if k == i or k == j:
                pauli_string += "Z"
            else:
                pauli_string += "I"
    return pauli_string


def get_weighted_mean(coreset_vectors, coreset_weights):
    dim = len(coreset_vectors[0])
    size_coreset = len(coreset_vectors)
    mu = np.zeros(dim)
    for i in range(size_coreset):
        mu += coreset_vectors[i] * coreset_weights[i]
    return mu / sum(coreset_weights)


def get_weighted_scatter_matrix(coreset, weights):
    dim = len(coreset[0])
    size_coreset = len(coreset)
    T = np.zeros((dim, dim))
    mu = get_weighted_mean(coreset, weights)
    for i in range(size_coreset):
        T += weights[i] * np.outer((coreset[i] - mu), (coreset[i] - mu))
    return T


def create_pauli_operators(coreset_vectors, coreset_weights):
    paulis = []
    pauli_weights = []

    T_inv = inv(get_weighted_scatter_matrix(coreset_vectors, coreset_weights))

    W = sum(coreset_weights)

    for i in range(len(coreset_vectors)):
        paulis += [Z_i(-1, len(coreset_vectors))]
        pauli_weights += [
            coreset_weights[i] ** 2
            * np.dot(coreset_vectors[i], np.dot(T_inv, coreset_vectors[i]))
        ]

        for l in range(len(coreset_vectors)):
            paulis += [Z_ij(i, l, len(coreset_vectors))]
            pauli_weights += [
                -2
                * coreset_weights[l]
                * coreset_weights[i] ** 2
                * np.dot(coreset_vectors[i], np.dot(T_inv, coreset_vectors[i]))
                / W
            ]

    for j in range(len(coreset_vectors)):
        for i in range(j):
            paulis += [Z_ij(i, j, len(coreset_vectors))]
            pauli_weights += [
                2
                * coreset_weights[i]
                * coreset_weights[j]
                * np.dot(coreset_vectors[i], np.dot(T_inv, coreset_vectors[j]))
            ]
            for l in range(len(coreset_vectors)):
                paulis += [Z_ij(i, l, len(coreset_vectors))]
                pauli_weights += [
                    -2
                    * coreset_weights[l]
                    * coreset_weights[i]
                    * coreset_weights[j]
                    * np.dot(coreset_vectors[i], np.dot(T_inv, coreset_vectors[j]))
                    / W
                ]
                paulis += [Z_ij(j, l, len(coreset_vectors))]
                pauli_weights += [
                    -2
                    * coreset_weights[l]
                    * coreset_weights[i]
                    * coreset_weights[j]
                    * np.dot(coreset_vectors[i], np.dot(T_inv, coreset_vectors[j]))
                    / W
                ]

    return [([pauli, weight]) for pauli, weight in zip(paulis, pauli_weights)]


def create_GMM_hamiltonian(coreset_vectors, coreset_weights):
    pauli_operators = create_pauli_operators(coreset_vectors, coreset_weights)
    H = 0
    for idx, op in enumerate(pauli_operators):
        operator_string = op[0]
        coeff = op[1]
        operator = 1
        print(f"idx :{idx}, op_string: {operator_string}")

        for i in range(len(operator_string)):
            op_i = operator_string[i]
            if op_i == "Z":
                operator *= spin.z(i)
            if op_i == "I":
                operator *= spin.i(i)
        print(operator)
        H += coeff * operator

    return -1 * H


def run_gmm(
    raw_data,
    number_of_qubits,
    number_of_centroid_evaluation,
    number_of_corsets_to_evaluate,
    circuit_depth,
    max_shots,
    max_iterations,
    normalize=True,
    centralize=True,
):
    coreset = Coreset()

    coreset_vectors, coreset_weights = coreset.get_best_coresets(
        data_vectors=raw_data,
        number_of_runs=number_of_centroid_evaluation,
        coreset_numbers=number_of_qubits,
        size_vec_list=number_of_corsets_to_evaluate,
    )

    if normalize:
        coreset_vectors, coreset_weights = normalize_np(
            coreset_vectors, centralize=centralize
        ), normalize_np(coreset_weights, centralize=centralize)

    return get_gmm_bitstring(
        coreset_vectors,
        coreset_weights,
        number_of_qubits,
        circuit_depth,
        max_shots,
        max_iterations,
    )


def get_gmm_bitstring(
    coreset_vectors,
    coreset_weights,
    number_of_qubits,
    circuit_depth,
    max_shots,
    max_iterations,
):
    Hamiltonian = create_GMM_hamiltonian(coreset_vectors, coreset_weights)

    optimizer, parameter_count = get_optimizer(
        max_iterations, circuit_depth, number_of_qubits
    )
    optimal_expectation, optimal_parameters = cudaq.vqe(
        kernel=kernel_two_local(number_of_qubits, circuit_depth),
        spin_operator=Hamiltonian[0],
        optimizer=optimizer,
        parameter_count=parameter_count,
        shots=max_shots,
    )

    counts = cudaq.sample(
        kernel_two_local(number_of_qubits, circuit_depth),
        optimal_parameters,
        shots_count=max_shots,
    )

    return counts.most_probable()
