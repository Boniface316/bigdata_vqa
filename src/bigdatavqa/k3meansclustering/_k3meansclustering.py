from abc import abstractmethod

from .._base import BigDataVQA
from ..coreset import Coreset

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


class K3MeansClustering(BigDataVQA):
    def __init__(
        self,
        full_coreset_df,
        vector_columns,
        weights_columns,
        normalize_vectors: bool = True,
        number_of_qubits_representing_data: int = 2,
    ) -> None:
        super().__init__(
            full_coreset_df,
            vector_columns,
            weights_columns,
            normalize_vectors,
            number_of_qubits_representing_data,
        )

    @abstractmethod
    def run_k3_clustering(self):
        pass

    @abstractmethod
    def _get_best_bitstring(self, coreset_graph):
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

        return [s1, s2, s3]

    # @staticmethod
    # def get_3means_cost(data_sets, cluster_centers):
    #     cost = 0
    #     labels = []

    #     for row_data in data_sets:
    #         dist = []
    #         for center in cluster_centers:
    #             dist.append(np.linalg.norm(row_data - center) ** 2)

    #         labels.append(np.argmin(dist))
    #         cost += min(dist)

    #     return cost, labels

    def fit(self):
        self.partition = self.run_k3_clustering()
        self.cluster_centers = self._get_cluster_centers_from_partition(self.partition)
        self.labels = self.get_lables_from_partition(
            self.full_coreset_df, self.partition
        )

    def get_lables_from_partition(self, coreset_df, partition):
        for i, cluster in enumerate(partition):
            for vertex in cluster:
                coreset_df.loc[int(vertex / 2), "label"] = i

        return coreset_df.label

    def _get_cluster_centers_from_partition(self, partition):
        coreset_vectors, coreset_weights = self.preprocess_data(
            self.full_coreset_df, self.vector_columns, self.weights_column, False
        )

        cluster_size = len(self.vector_columns)
        clusters_centers = np.array([np.zeros(cluster_size)] * 3)

        W = np.sum(coreset_weights) / 3

        for i in range(len(partition)):
            for vertex in partition[i]:
                weight = (
                    coreset_weights[
                        int(vertex / self.number_of_qubits_representing_data)
                    ]
                    * coreset_vectors[
                        int(vertex / self.number_of_qubits_representing_data)
                    ]
                )
                clusters_centers[i] += weight * (1 / W)

        return clusters_centers


class K3MeansClusteringVQA(K3MeansClustering):
    def __init__(
        self,
        full_coreset_df,
        vector_columns,
        weights_columns,
        qubits,
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
        super().__init__(
            full_coreset_df,
            vector_columns,
            weights_columns,
            normalize_vectors,
            number_of_qubits_representing_data,
        )
        self.qubits = qubits
        self.create_Hamiltonian = create_Hamiltonian
        self.optimizer_function = optimizer_function
        self.optimizer = optimizer
        self.create_circuit = create_circuit
        self.circuit_depth = circuit_depth
        self.max_iterations = max_iterations
        self.max_shots = max_shots
        self.coreset_to_graph_metric = coreset_to_graph_metric

    def run_k3_clustering(self):
        coreset_vectors, coreset_weights = self.preprocess_data(
            self.full_coreset_df,
            self.vector_columns,
            self.weights_column,
            self.normalize_vectors,
        )

        G = Coreset.coreset_to_graph(
            coreset_vectors,
            coreset_weights,
            self.coreset_to_graph_metric,
            self.number_of_qubits_representing_data,
        )

        Hamiltonian = self.create_Hamiltonian(G)

        optimizer, parameter_count = self.optimizer_function(
            self.optimizer,
            self.max_iterations,
            qubits=self.qubits,
            circuit_depth=self.circuit_depth,
        )

        kernel = self.create_circuit(self.qubits, self.circuit_depth)

        counts = self.get_counts(
            self.qubits, Hamiltonian, kernel, optimizer, parameter_count
        )

        self.bitstring = self._get_best_bitstring(counts)

        return self.get_partition_from_bitstring(self.bitstring, G)

    def _get_best_bitstring(self, counts):
        return counts.most_probable()


class K3MeansClusteringRandom(K3MeansClustering):
    def __init__(
        self,
        normalize_vectors: bool = True,
        number_of_qubits_representing_data: int = 2,
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
        bitstring_length = (
            coreset_graph.number_of_nodes() * self.number_of_qubits_representing_data
        )

        bitstring_not_accepted = True
        while bitstring_not_accepted:
            bitstring = np.random.randint(0, 2, bitstring_length)
            if bitstring.sum() == 0 or bitstring.sum() == len(bitstring):
                bitstring_not_accepted = True
            else:
                bitstring_not_accepted = False

        return "".join([str(i) for i in bitstring])


class K3MeansClusteringMaxCut(K3MeansClustering):
    def __init__(
        self,
        normalize_vectors: bool = True,
        number_of_qubits_representing_data: int = 2,
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

        self.get_optimal_bitstring(
            coreset_graph, coreset_df, vector_columns, weight_columns
        )

        return self.best_centers

    def get_optimal_bitstring(
        self, coreset_graph, coreset_df, vector_columns, weight_columns
    ):
        bitstring_length = (
            coreset_graph.number_of_nodes() * self.number_of_qubits_representing_data
        )

        bitstrings = self.create_all_possible_bitstrings(bitstring_length)

        lowest_cost = np.inf
        best_bitstring = None
        self.best_centers = None

        for bitstring in tqdm(bitstrings):
            current_bitstring_cost = 0
            partitions = self.get_partition_from_bitstring(bitstring, coreset_graph)
            centers = self.get_cluster_centers_from_partition(
                coreset_df, partitions, vector_columns
            )
            for k, grouped_data in coreset_df.groupby("k"):
                current_bitstring_cost += self.get_3means_cost(
                    grouped_data[vector_columns].to_numpy(), [centers[k]]
                )[0]

            if current_bitstring_cost < lowest_cost:
                lowest_cost = current_bitstring_cost
                best_bitstring = bitstring
                self.best_centers = centers

        return best_bitstring

    def create_all_possible_bitstrings(self, bitstring_length):
        return [
            format(i, f"0{bitstring_length}b")
            for i in range(1, (2**bitstring_length) - 1)
        ]

    def get_cluster_centers_from_partition(
        self, coreset_df, partitions, vector_columns
    ):
        coreset_df["k"] = None

        partitions = [partition for partition in partitions if partition]

        for k, partition in enumerate(partitions):
            index_values = np.array(partition) / 2
            coreset_df.loc[index_values, "k"] = k

        columns_to_include = vector_columns + ["k"]

        return coreset_df[columns_to_include].groupby("k").mean().values


class K3MeansClusteringKMeans(K3MeansClustering):
    def __init__(
        self,
        normalize_vectors: bool = True,
        number_of_qubits_representing_data: int = 2,
    ):
        super().__init__(normalize_vectors, number_of_qubits_representing_data)

    def get_cluster_centers(self, coreset_df, vector_columns, weight_columns):
        data_set = coreset_df.copy()
        data_set["weights"] = 0
        coreset_vectors, _ = self.preprocess_data(
            data_set, vector_columns, weight_columns, self.normalize_vectors
        )

        return KMeans(n_clusters=3).fit(coreset_vectors).cluster_centers_

    def get_optimal_bitstring(self, coreset_graph):
        return None
