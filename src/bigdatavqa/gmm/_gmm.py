from abc import ABC, abstractmethod

import cudaq
import numpy as np
from cudaq import spin
from numpy.linalg import inv


class GMMClustering(ABC):
    def __init__(normalize_vectors=True):
        self.normalize_vectors = normalize_vectors

    def preprocess_data(self, corese_df, vector_columns, weight_columns, normalize_vectors=True):
        coreset_vectors = corese_df[vector_columns].to_numpy()
        coreset_weights = corese_df[weight_columns].to_numpy()

        if normalize_vectors:
            coreset_vectors = Coreset.normalize_array(coreset_vectors, True)
            coreset_weights = Coreset.normalize_array(coreset_weights)

        return coreset_vectors, coreset_weights

    def run_clustering(self, coreset_df, vector_columns, weight_columns):
        coreset_vectors, coreset_weights = self.preprocess_data(
            coreset_df, vector_columns, weight_columns, self.normalize_vectors
        )

        self.counts = self.get_labels(coreset_vectors, coreset_weights)
        self.cost = self.get_cost(coreset_vectors, coreset_weights)

    @abstractmethod
    def get_labels(self, counts):
        pass


class GMMClusteringVQA(GMMclustering):
    def __init__(
        self,
        qubits,
        create_circuit,
        circuit_depth,
        optimizer_function,
        optimizer,
        create_Hamiltonian,
        max_iterations,
        max_shots,
        normalize_vectors=True,
    ):
        super().__init__(normalize_vectors)
        self.qubits = qubits
        self.create_circuit = create_circuit
        self.circuit_depth = circuit_depth
        self.optimizer_function = optimizer_function
        self.optimizer = optimizer
        self.create_Hamiltonian = create_Hamiltonian
        self.max_iterations = max_iterations
        self.max_shots = max_shots

    def Z_i(self, i, length):
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

    def Z_ij(self, i, j, length):
        pauli_string = ""
        if i == j:
            pauli_string = self.Z_i(-1, length)
        else:
            for k in range(length):
                if k == i or k == j:
                    pauli_string += "Z"
                else:
                    pauli_string += "I"
        return pauli_string

    def get_scatter_matrix(self, coreset_vectors, coreset_weights=None):
        coreset_size, columns = coreset_vectors.shape
        if coreset_weights is None:
            coreset_weights = np.ones(coreset_size)
        T = np.zeros((columns, columns))
        mu = np.average(coreset_vectors, axis=0, weights=coreset_weights)
        for i in range(coreset_size):
            T += coreset_weights[i] * np.outer((coreset_vectors[i] - mu), (coreset_vectors[i] - mu))
        return T

    def create_pauli_operators(self, coreset_vectors, coreset_weights):
        paulis = []
        pauli_weights = []

        T = self.get_scatter_matrix(coreset_vectors, coreset_weights)

        T_inv = inv(T)

        W = sum(coreset_weights)

        for i in range(self.qubits):
            paulis += [self.Z_i(-1, self.qubits)]
            pauli_weights += [
                coreset_weights[i] ** 2
                * np.dot(coreset_vectors[i], np.dot(T_inv, coreset_vectors[i]))
            ]

            for l in range(self.qubits):
                paulis += [self.Z_ij(i, l, self.qubits)]
                pauli_weights += [
                    -2
                    * coreset_weights[l]
                    * coreset_weights[i] ** 2
                    * np.dot(coreset_vectors[i], np.dot(T_inv, coreset_vectors[i]))
                    / W
                ]

        for j in range(self.qubits):
            for i in range(j):
                paulis += [self.Z_ij(i, j, self.qubits)]
                pauli_weights += [
                    2
                    * coreset_weights[i]
                    * coreset_weights[j]
                    * np.dot(coreset_vectors[i], np.dot(T_inv, coreset_vectors[j]))
                ]
                for l in range(self.qubits):
                    paulis += [self.Z_ij(i, l, self.qubits)]
                    pauli_weights += [
                        -2
                        * coreset_weights[l]
                        * coreset_weights[i]
                        * coreset_weights[j]
                        * np.dot(coreset_vectors[i], np.dot(T_inv, coreset_vectors[j]))
                        / W
                    ]
                    paulis += [self.Z_ij(j, l, self.qubits)]
                    pauli_weights += [
                        -2
                        * coreset_weights[l]
                        * coreset_weights[i]
                        * coreset_weights[j]
                        * np.dot(coreset_vectors[i], np.dot(T_inv, coreset_vectors[j]))
                        / W
                    ]

        return [([pauli, weight]) for pauli, weight in zip(paulis, pauli_weights)]

    def get_labels(self, coreset_vectors, coreset_weights):
        optimizer, parameter_count = self.optimizer_function(
            self.optimizer,
            self.max_iterations,
            qubits=self.qubits,
            circuit_depth=self.circuit_depth,
        )

        pauli_operators = self.create_pauli_operators(coreset_vectors, coreset_weights)

        kernel = self.create_circuit(self.qubits, self.circuit_depth)

        Hamiltonian = self.create_Hamiltonian(pauli_operators)

        def objective_function(
            parameter_vector: list[float],
            hamiltonian=Hamiltonian,
            kernel=kernel,
        ) -> tuple[float, list[float]]:
            get_result = lambda parameter_vector: cudaq.observe(
                kernel, hamiltonian, parameter_vector, self.qubits, self.circuit_depth
            ).expectation()

            cost = get_result(parameter_vector)

            return cost

        self.energy, self.optimal_parameters = optimizer.optimize(
            dimensions=parameter_count, function=objective_function
        )

        self.counts = cudaq.sample(
            kernel,
            self.optimal_parameters,
            self.qubits,
            self.circuit_depth,
            shots_count=self.max_shots,
        )

        return self.counts.most_probable()


class GMMClusteringRandom(GMMClustering):
    def __init__(self, normalize_vectors=True):
        super().__init__(normalize_vectors)

    def get_labels(self, coreset_vectors, coreset_weights):
        bitstring_length = len(coreset_vectors)

        bitstring_not_accepted = True
        while bitstring_not_accepted:
            bitstring = np.random.randint(0, 2, bitstring_length)
            if bitstring.sum() == 0 or bitstring.sum() == len(bitstring):
                bitstring_not_accepted = True
            else:
                bitstring_not_accepted = False

        return "".join([str(i) for i in bitstring])


class GMMClusteringMaxCut(GMMClustering):
    def __init__(self, normalize_vectors=True):
        super().__init__(normalize_vectors)

    def get_labels(self, coreset_vectors, coreset_weights):
        pass


class GMMClusteringKMeans(GMMClustering):
    def __init__(self, normalize_vectors=True):
        super().__init__(normalize_vectors)

    def get_labels(self, coreset_vectors, coreset_weights):
        pass
