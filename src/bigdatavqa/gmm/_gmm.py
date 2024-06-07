from abc import ABC, abstractmethod

from bigdatavqa.coreset import Coreset

import cudaq
import numpy as np
from cudaq import spin
from numpy.linalg import inv
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


class GMMClustering(ABC):
    def __init__(self, normalize_vectors=True):
        self.normalize_vectors = normalize_vectors
        self.cost = None

    def preprocess_data(
        self, coreset_df, vector_columns, weight_columns, normalize_vectors=True
    ):
        coreset_vectors = coreset_df[vector_columns].to_numpy()
        coreset_weights = coreset_df[weight_columns].to_numpy()

        if normalize_vectors:
            coreset_vectors = Coreset.normalize_array(coreset_vectors, True)
            coreset_weights = Coreset.normalize_array(coreset_weights)

        return coreset_vectors, coreset_weights

    def fit(self, coreset_df, vector_columns, weight_columns):
        coreset_vectors, coreset_weights = self.preprocess_data(
            coreset_df, vector_columns, weight_columns, self.normalize_vectors
        )

        self.labels = self.get_labels(
            coreset_df=coreset_df,
            coreset_vectors=coreset_vectors,
            coreset_weights=coreset_weights,
            vector_columns=vector_columns,
            weight_columns=weight_columns,
        )
        if self.cost is None:
            coreset_vectors_origial = coreset_df[vector_columns].to_numpy()
            self.cost, _ = self.get_centroids_based_cost(
                coreset_vectors_origial, self.cluster_centers
            )

    @abstractmethod
    def get_labels(
        self,
        coreset_df,
        coreset_vectors,
        coreset_weights,
        vector_columns,
        weight_columns,
    ):
        pass

    @staticmethod
    def get_centroids_based_cost(data_sets, cluster_centers):
        cost = 0
        labels = []

        for row_data in data_sets:
            dist = []
            for center in cluster_centers:
                dist.append(np.linalg.norm(row_data - center) ** 2)

            labels.append(np.argmin(dist))
            cost += min(dist)

        return cost, labels

    def get_cluster_centroids_from_bitstring(self, coreset_df, vector_columns):
        columns_retain = vector_columns + ["k"]

        return coreset_df[columns_retain].groupby("k").mean().values


class GMMClusteringVQA(GMMClustering):
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
            T += coreset_weights[i] * np.outer(
                (coreset_vectors[i] - mu), (coreset_vectors[i] - mu)
            )
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

    def get_labels(
        self,
        coreset_df,
        coreset_vectors,
        coreset_weights,
        vector_columns,
        *args,
        **kwargs,
    ):
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

        bitstring = self.counts.most_probable()
        bitstring = [int(i) for i in bitstring]
        coreset_df["k"] = bitstring
        self.cluster_centers = self.get_cluster_centroids_from_bitstring(
            coreset_df, vector_columns
        )

        return bitstring


class GMMClusteringRandom(GMMClustering):
    def __init__(self, normalize_vectors=True):
        super().__init__(normalize_vectors)

    def get_labels(self, coreset_df, coreset_vectors, vector_columns, *args, **kwargs):
        bitstring_length = len(coreset_vectors)

        bitstring_not_accepted = True
        while bitstring_not_accepted:
            bitstring = np.random.randint(0, 2, bitstring_length)
            if bitstring.sum() == 0 or bitstring.sum() == len(bitstring):
                bitstring_not_accepted = True
            else:
                bitstring_not_accepted = False

        bitstring = [int(i) for i in bitstring]
        coreset_df["k"] = bitstring
        self.cluster_centers = self.get_cluster_centroids_from_bitstring(
            coreset_df, vector_columns
        )

        return [int(i) for i in bitstring]


class GMMClusteringMaxCut(GMMClustering):
    def __init__(self, normalize_vectors=True):
        super().__init__(normalize_vectors)

    def get_labels(
        self,
        coreset_df,
        coreset_vectors,
        coreset_weights,
        vector_columns,
        weight_columns,
    ):
        bitstring_length = len(coreset_vectors)
        bitstrings = self.create_all_possible_bitstrings(bitstring_length)

        self.cost = np.inf
        best_bitstring = None
        self.cluster_centers = None

        for bitstring in tqdm(bitstrings):
            bitstring = [int(i) for i in bitstring]
            current_bitstring_cost, cluster_centers = (
                self._get_bistring_cost_and_centroids(
                    coreset_df, bitstring, vector_columns
                )
            )

            if current_bitstring_cost < self.cost:
                best_bitstring = bitstring
                self.cost = current_bitstring_cost
                self.cluster_centers = cluster_centers

        return best_bitstring

    def _get_bistring_cost_and_centroids(self, coreset_df, bitstring, vector_columns):
        coreset_df["k"] = bitstring
        current_bitstring_cost = 0
        centroids = self.get_cluster_centroids_from_bitstring(
            coreset_df, vector_columns
        )

        for k, grouped_data in coreset_df.groupby("k"):
            grouped_data_vector = grouped_data[vector_columns].to_numpy()

            current_bitstring_cost += self.get_centroids_based_cost(
                grouped_data_vector, [centroids[k]]
            )[0]

        return current_bitstring_cost, centroids

    def create_all_possible_bitstrings(self, bitstring_length):
        return [
            format(i, f"0{bitstring_length}b")
            for i in range(1, (2**bitstring_length) - 1)
        ]


class GMMClusteringClassicalGMM(GMMClustering):
    def __init__(self, normalize_vectors=True):
        super().__init__(normalize_vectors)

    def get_labels(
        self,
        coreset_df,
        coreset_vectors,
        vector_columns,
        weight_columns,
        n_components=2,
        *args,
        **kwargs,
    ):
        gmm = GaussianMixture(n_components=n_components)
        labels = gmm.fit_predict(coreset_vectors)
        coreset_df["k"] = labels

        self.cluster_centers = self.get_cluster_centroids_from_bitstring(
            coreset_df, vector_columns
        )

        return labels
