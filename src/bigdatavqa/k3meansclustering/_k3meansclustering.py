import random
from abc import ABC, abstractmethod

import cudaq
import numpy as np
import pandas as pd

from ..coreset import Coreset


class K3MeansClustering(ABC):
    def __init__(
        self, normalize_vectors: bool = True, number_of_qubits_representing_data: int = 2
    ) -> None:
        self.normalize_vectors = normalize_vectors
        self.number_of_qubits_representing_data = number_of_qubits_representing_data

    @abstractmethod
    def get_cluster_centers(self, coreset_df, vector_columns, weight_columns):
        pass

    @abstractmethod
    def get_optimal_bitstring(self, coreset_graph):
        pass

    def get_partition_from_bitstring(self, bitstring, coreset_graph):
        s1, s2, s3 = [], [], []

        vertices = list(coreset_graph.nodes)

        pairs = [bitstring[i : i + 2] for i in range(0, len(bitstring), 2)]
        for i, vertex in enumerate(pairs):
            if vertex == "00":
                s1.append(vertices[i])
            elif vertex == "10":
                s2.append(vertices[i])
            else:
                s3.append(vertices[i])

        # TODO: verify the outcome

        return [s1, s2, s3]

    def preprocess_data(self, corese_df, vector_columns, weight_columns, normalize_vectors=True):
        coreset_vectors = corese_df[vector_columns].to_numpy()
        coreset_weights = corese_df[weight_columns].to_numpy()

        if normalize_vectors:
            coreset_vectors = Coreset.normalize_array(coreset_vectors, True)
            coreset_weights = Coreset.normalize_array(coreset_weights)

        return coreset_vectors, coreset_weights

    @staticmethod
    def get_3means_cost(raw_data, cluster_centers):
        center1, center2, center3 = cluster_centers
        cost = 0

        for row_data in raw_data:
            dist = []
            dist.append(np.linalg.norm(row_data - center1) ** 2)
            dist.append(np.linalg.norm(row_data - center2) ** 2)
            dist.append(np.linalg.norm(row_data - center3) ** 2)
            cost += min(dist)

        return cost

    def get_cluster_centers_from_partition(
        self, partition, coreset_df, vector_columns, weight_columns
    ):
        coreset_vectors, coreset_weights = self.preprocess_data(
            coreset_df, vector_columns, weight_columns, False
        )

        cluster_size = len(vector_columns)
        clusters_centers = np.array([np.zeros(cluster_size)] * 3)

        W = np.sum(coreset_weights) / 3

        for i in range(len(partition)):
            for vertex in partition[i]:
                weight = (
                    coreset_weights[int(vertex / self.number_of_qubits_representing_data)]
                    * coreset_vectors[int(vertex / self.number_of_qubits_representing_data)]
                )
                clusters_centers[i] += weight * (1 / W)

        return clusters_centers


class K3MeansClusteringVQE(K3MeansClustering):
    def __init__(
        self,
        create_circuit,
        circuit_depth,
        optimizer_function,
        optimizer,
        create_Hamiltonian,
        number_of_qubits_representing_data,
        normalize_vectors,
        max_iterations,
        max_shots,
        coreset_to_graph_metric="dist",
    ) -> None:
        super().__init__(normalize_vectors, number_of_qubits_representing_data)
        self.create_Hamiltonian = create_Hamiltonian
        self.optimizer_function = optimizer_function
        self.optimizer = optimizer
        self.create_circuit = create_circuit
        self.circuit_depth = circuit_depth
        self.max_iterations = max_iterations
        self.max_shots = max_shots
        self.coreset_to_graph_metric = coreset_to_graph_metric

    def get_cluster_centers(self, coreset_df, vector_columns, weight_columns):
        coreset_vectors, coreset_weights = self.preprocess_data(
            coreset_df, vector_columns, weight_columns, self.normalize_vectors
        )

        coreset_graph = Coreset.coreset_to_graph(
            coreset_vectors,
            coreset_weights,
            self.coreset_to_graph_metric,
            self.number_of_qubits_representing_data,
        )

        bitstring = self.get_optimal_bitstring(coreset_graph)

        partition = self.get_partition_from_bitstring(bitstring)

        return self.get_cluster_centers_from_partition(
            partition, coreset_df, vector_columns, weight_columns
        )

    def get_optimal_bitstring(self, coreset_graph):
        number_of_qubits = self.number_of_qubits_representing_data * coreset_graph.number_of_nodes()

        Hamiltonian = self.create_Hamiltonian(coreset_graph)
        optimizer, parameter_count = self.optimizer_function(
            self.optimizer,
            self.max_iterations,
            qubits=number_of_qubits,
            circuit_depth=self.circuit_depth,
        )

        kernel = self.create_circuit(number_of_qubits, self.circuit_depth)

        def objective_function(
            parameter_vector: list[float],
            hamiltonian=Hamiltonian,
            kernel=kernel,
        ) -> tuple[float, list[float]]:
            get_result = lambda parameter_vector: cudaq.observe(
                kernel, hamiltonian, parameter_vector, qubits, circuit_depth
            ).expectation()

            cost = get_result(parameter_vector)

            return cost

        energy, optimal_parameters = optimizer.optimize(
            dimensions=parameter_count, function=objective_function
        )

        counts = cudaq.sample(
            kernel, optimal_parameters, qubits, circuit_depth, shots_count=max_shots
        )

        return counts.most_probable()


class K3MeansClusteringRandom(K3MeansClustering):
    def __init__(
        self, normalize_vectors: bool = True, number_of_qubits_representing_data: int = 2
    ) -> None:
        super().__init__(normalize_vectors, number_of_qubits_representing_data)

    def get_cluster_centers(self, coreset_df, vector_columns, weight_columns):
        coreset_vectors, coreset_weights = self.preprocess_data(
            coreset_df, vector_columns, weight_columns, self.normalize_vectors
        )

        coreset_graph = Coreset.coreset_to_graph(
            coreset_vectors,
            coreset_weights,
            "dist",
            self.number_of_qubits_representing_data,
        )

        bitstring = self.get_optimal_bitstring(coreset_graph)

        partition = self.get_partition_from_bitstring(bitstring, coreset_graph)

        return self.get_cluster_centers_from_partition(
            partition, coreset_df, vector_columns, weight_columns
        )

    def get_optimal_bitstring(self, coreset_graph):
        bitstring_length = coreset_graph.number_of_nodes() * self.number_of_qubits_representing_data

        bitstring_not_accepted = True
        while bitstring_not_accepted:
            bitstring = np.random.randint(0, 2, bitstring_length)
            if bitstring.sum() == 0 or bitstring.sum() == len(bitstring):
                bitstring_not_accepted = True
            else:
                bitstring_not_accepted = False

        return bitstring


class K3MeansClusteringMaxCut(K3MeansClustering):
    def __init__(
        self, normalize_vectors: bool = True, number_of_qubits_representing_data: int = 2
    ) -> None:
        super().__init__(normalize_vectors, number_of_qubits_representing_data)

    def get_cluster_centers(self, coreset_df, vector_columns, weight_columns):
        coreset_vectors, coreset_weights = self.preprocess_data(
            coreset_df, vector_columns, weight_columns, self.normalize_vectors
        )

        coreset_graph = Coreset.coreset_to_graph(
            coreset_vectors,
            coreset_weights,
            "dist",
            self.number_of_qubits_representing_data,
        )

        bitstring = self.get_optimal_bitstring(coreset_graph)

        partition = self.get_partition_from_bitstring(bitstring, coreset_graph)

        return self.get_cluster_centers_from_partition(
            partition, coreset_df, vector_columns, weight_columns
        )

    def get_optimal_bitstring(self, coreset_graph, coreset_df, vector_columns, weight_columns):
        bitstring_length = coreset_graph.number_of_nodes() * self.number_of_qubits_representing_data

        bitstrings = self.create_all_possible_bitstrings(bitstring_length)

        lowest_cost = np.inf
        best_bitstring = None

        for bitstring in bitstrings:
            partition = self.get_partition_from_bitstring(bitstring, coreset_graph)
            current_cost = self._get_brute_force_cost(
                coreset_df, vector_columns, weight_columns, partition
            )
            if current_cost < lowest_cost:
                lowest_cost = current_cost
                best_bitstring = bitstring

        return best_bitstring

    def create_all_possible_bitstrings(self, bitstring_length):
        return [format(i, f"0{bitstring_length}b") for i in range(1, (2**bitstring_length) - 1)]

    def get_distance_between_two_vectors(
        self, vector1: np.ndarray, vector2: np.ndarray, weight: np.ndarray
    ) -> float:
        return weight * np.linalg.norm(vector1 - vector2)

    def get_weight_k3_means_cost(self, k, cluster_centers, data, data_weights=None):
        accumulativeCost = 0
        currentCosts = np.repeat(0, k)
        data_weights = np.repeat(1, len(data)) if data_weights is None else data_weights
        for vector in data:
            currentCosts = list(
                map(
                    self.get_distance_between_two_vectors,
                    cluster_centers,
                    np.repeat(vector, k, axis=0),
                    data_weights,
                )
            )
            accumulativeCost = accumulativeCost + min(currentCosts)

        return accumulativeCost

    def _get_brute_force_cost(self, coreset_vectors, coreset_weights, partition):
        coreset_partition = [[], [], []]
        weight_partition = [[], [], []]
        cluster_centres = np.empty((len(partition), 2))
        for i, subset in enumerate(partition):
            for index in subset:
                coreset_partition[i].append(coreset_vectors[int(index) - 1])
                weight_partition[i].append(coreset_weights[int(index) - 1])

        for i in range(len(partition)):
            cluster_centres[i] = np.average(
                coreset_partition[i], axis=0, weights=weight_partition[i]
            )

        return self.get_weight_k3_means_cost(3, cluster_centres, coreset_vectors, coreset_weights)
