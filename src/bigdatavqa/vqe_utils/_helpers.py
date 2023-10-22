from numpy.linalg import inv
import numpy as np


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
