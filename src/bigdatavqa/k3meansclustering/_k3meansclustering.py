from abc import abstractmethod

from .._base import BigDataVQA
from ..coreset import Coreset

import matplotlib.pyplot as plt
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
        self.cluster_centers = None
        self.labels = None
        self.cost = None

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

    def fit(self):
        self.partition = self.run_k3_clustering()
        self.partition = [
            [int(x / self.number_of_qubits_representing_data) for x in sublist]
            for sublist in self.partition
        ]
        if self.cluster_centers is None:
            self.cluster_centers = self._get_cluster_centers_from_partition(
                self.partition
            )
        if self.labels is None:
            self.labels = self.get_lables_from_partition(
                self.full_coreset_df, self.partition
            )
            self.full_coreset_df["label"] = self.labels
        if self.cost is None:
            self.cost = self.get_cost_using_kmeans_approach(
                self.full_coreset_df, self.cluster_centers
            )

    def get_lables_from_partition(self, coreset_df, partition):
        labels = [0] * len(coreset_df)
        for i, cluster in enumerate(partition):
            for vertex in cluster:
                labels[vertex] = i

        return labels

    def _get_cluster_centers_from_partition(self, partition):
        coreset_vectors, coreset_weights = self.preprocess_data(
            self.full_coreset_df, self.vector_columns, self.weights_column, False
        )

        cluster_size = len(self.vector_columns)
        clusters_centers = np.array([np.zeros(cluster_size)] * 3)

        W = np.sum(coreset_weights) / 3

        for cluster, vertices in enumerate(partition):
            for vertex in vertices:
                weight = coreset_weights[vertex] * coreset_vectors[vertex]
                clusters_centers[cluster] += weight * (1 / W)

        return clusters_centers

    def plot(self):
        plt.scatter(
            self.full_coreset_df["X"],
            self.full_coreset_df["Y"],
            c=self.full_coreset_df["label"],
            label="Coreset",
            cmap="viridis",
        )
        plt.scatter(
            self.cluster_centers[:, 0],
            self.cluster_centers[:, 1],
            label="Centers",
            color="b",
            marker="*",
        )

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Raw data and its best coreset using BFL16")
        plt.legend()
        plt.show()


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
            "dist",
            self.number_of_qubits_representing_data,
        )

        bitstring = self.generate_random_bitstring(coreset_vectors)
        bitstring = "".join([str(x) for x in bitstring])

        return self.get_partition_from_bitstring(bitstring, G)

    def _get_best_bitstring(self):
        pass


class K3MeansClusteringKMeans(K3MeansClustering):
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

    def run_k3_clustering(self):
        coreset_vectors, _ = self.preprocess_data(
            self.full_coreset_df,
            self.vector_columns,
            self.weights_column,
            self.normalize_vectors,
        )

        self._KMeans = KMeans(n_clusters=3).fit(coreset_vectors)

        return [
            list(
                np.where(self._KMeans.labels_ == i)[0]
                * self.number_of_qubits_representing_data
            )
            for i in range(3)
        ]

    def _get_best_bitstring(self, coreset_graph):
        pass


class K3MeansClusteringMaxCut(K3MeansClustering):
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

    def run_k3_clustering(self):
        coreset_vectors, coreset_weights = self.preprocess_data(
            self.full_coreset_df,
            self.vector_columns,
            self.weights_column,
            self.normalize_vectors,
        )

        coreset_graph = Coreset.coreset_to_graph(
            coreset_vectors,
            coreset_weights,
            "dist",
            self.number_of_qubits_representing_data,
        )

        bitstring = self._get_best_bitstring(
            coreset_graph,
        )

        return self.get_partition_from_bitstring(bitstring, coreset_graph)

    def _get_best_bitstring(self, coreset_graph):
        coreset_df = self.full_coreset_df.copy()

        bitstring_length = (
            coreset_graph.number_of_nodes() * self.number_of_qubits_representing_data
        )

        bitstrings = self.create_all_possible_bitstrings(bitstring_length)

        lowest_cost = np.inf
        best_bitstring = None

        for bitstring in tqdm(bitstrings):
            current_bitstring_cost = 0
            partition = self.get_partition_from_bitstring(bitstring, coreset_graph)
            partition = [
                [int(x / self.number_of_qubits_representing_data) for x in sublist]
                for sublist in partition
            ]
            cluster_centers = self._get_cluster_centers_from_partition(partition)

            labels = self.get_lables_from_partition(self.full_coreset_df, partition)

            coreset_df["label"] = labels
            current_bitstring_cost = self.get_cost_using_kmeans_approach(
                coreset_df, cluster_centers
            )

            if current_bitstring_cost < lowest_cost:
                lowest_cost = current_bitstring_cost
                best_bitstring = bitstring
                self.cluster_centers = cluster_centers
                self.labels = labels
                self.full_coreset_df["label"] = labels
                self.cost = current_bitstring_cost

        return best_bitstring
