from abc import abstractmethod
from typing import Callable, List, Optional

from .._base import BaseConfig, BigDataVQA, VQAConfig
from ..coreset import Coreset

import cudaq
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm


class K3MeansClustering(BigDataVQA):
    def __init__(
        self,
        full_coreset_df: pd.DataFrame,
        vector_columns: list[str],
        weights_column: list[str],
        normalize_vectors: Optional[bool] = True,
        number_of_qubits_representing_data: Optional[int] = 2,
    ) -> None:
        base_config = BaseConfig(
            vector_columns=vector_columns,
            weights_column=weights_column,
            normalize_vectors=normalize_vectors,
            number_of_qubits_representing_data=number_of_qubits_representing_data,
        )
        super().__init__(
            full_coreset_df=full_coreset_df,
            base_config=base_config,
        )
        self.cluster_centers = None
        self.labels = None
        self.cost = None

    @abstractmethod
    def run_k3_clustering(self) -> List:
        """
            This method should implement the logic to run the K3 clustering algorithm.

        Returns:
            List: The partition of the coreset into 3 clusters
        """

        pass

    @abstractmethod
    def _get_best_bitstring(self, coreset_graph: nx.Graph) -> str:
        """
        Returns the best bitstring
        """
        pass

    def get_partition_from_bitstring(
        self, bitstring: str, coreset_graph: nx.Graph
    ) -> List:
        """
        Converts the bitstring into a partition of the coreset into 3 clusters

        Args:
            bitstring (str): The bitstring
            coreset_graph (nx.Graph): The coreset graph

        Returns:
            List: The partition of the coreset into 3 clusters

        """
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
        """
        Fits the k3 means model to the data

        """

        self.partition = self.run_k3_clustering()
        self.partition = [
            [
                int(x / self.base_config.number_of_qubits_representing_data)
                for x in sublist
            ]
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

    def get_lables_from_partition(
        self, coreset_df: pd.DataFrame, partition: List
    ) -> List:
        """
        Converts the partition into labels for the coreset

        Args:
            coreset_df (pd.DataFrame): The coreset dataframe
            partition (List): The partition of the coreset

        Returns:
            List: The labels for the coreset
        """

        labels = [0] * len(coreset_df)
        for i, cluster in enumerate(partition):
            for vertex in cluster:
                labels[vertex] = i

        return labels

    def _get_cluster_centers_from_partition(self, partition: List) -> np.array:
        """
        Returns the cluster centers from the partition

        Args:
            partition (List): The partition of the coreset

        Returns:
            np.array: The cluster centers

        """

        coreset_vectors, coreset_weights = self.preprocess_data(self.full_coreset_df)

        cluster_size = len(self.base_config.vector_columns)
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
        full_coreset_df: pd.DataFrame,
        vector_columns: List[str],
        weights_column: str,
        qubits: int,
        create_circuit: Callable,
        circuit_depth: int,
        optimizer_function: Callable,
        optimizer: cudaq.optimizers.optimizer,
        create_Hamiltonian: cudaq.SpinOperator,
        number_of_qubits_representing_data: int,
        max_iterations: int,
        max_shots: int,
        coreset_to_graph_metric: Optional[str] = "dist",
        normalize_vectors: Optional[bool] = True,
    ) -> None:
        super().__init__(
            full_coreset_df,
            vector_columns,
            weights_column,
            normalize_vectors,
            number_of_qubits_representing_data,
        )

        self.VQA_config = VQAConfig(
            qubits=qubits,
            circuit_depth=circuit_depth,
            optimizer_function=optimizer_function,
            optimizer=optimizer,
            create_Hamiltonian=create_Hamiltonian,
            max_iterations=max_iterations,
            max_shots=max_shots,
            create_circuit=create_circuit,
            coreset_to_graph_metric=coreset_to_graph_metric,
        )

    def run_k3_clustering(self) -> List:
        """
        Run the K3 clustering algorithm using VQA

        Returns:
            List: The partition of the coreset into 3 clusters

        """

        coreset_vectors, coreset_weights = self.preprocess_data(self.full_coreset_df)

        G = Coreset.coreset_to_graph(
            coreset_vectors,
            coreset_weights,
            self.VQA_config.coreset_to_graph_metric,
            self.base_config.number_of_qubits_representing_data,
        )

        Hamiltonian = self.VQA_config.create_Hamiltonian(G)

        optimizer, parameter_count = self.VQA_config.optimizer_function(
            self.VQA_config.optimizer,
            self.VQA_config.max_iterations,
            qubits=self.VQA_config.qubits,
            circuit_depth=self.VQA_config.circuit_depth,
        )

        kernel = self.VQA_config.create_circuit(
            self.VQA_config.qubits, self.VQA_config.circuit_depth
        )

        counts = self.get_counts(
            self.VQA_config.qubits, Hamiltonian, kernel, optimizer, parameter_count
        )

        self.bitstring = self._get_best_bitstring(counts)

        return self.get_partition_from_bitstring(self.bitstring, G)

    def _get_best_bitstring(self, counts: cudaq.SampleResult):
        """
        Get the best bitstring from the counts object

        Args:
            counts (cudaq.SampleResult): The counts object from cudaq

        Returns:
            str: The best bitstring

        """
        return counts.most_probable()


class K3MeansClusteringRandom(K3MeansClustering):
    def __init__(
        self,
        full_coreset_df: pd.DataFrame,
        vector_columns: list[str],
        weights_column: list[str],
        normalize_vectors: bool = True,
        number_of_qubits_representing_data: int = 2,
    ) -> None:
        super().__init__(
            full_coreset_df,
            vector_columns,
            weights_column,
            normalize_vectors,
            number_of_qubits_representing_data,
        )

    def run_k3_clustering(self) -> List:
        """
        Runs the K3 clustering algorithm using a random bitstring

        Returns:
            List: The partition of the coreset into 3 clusters
        """

        coreset_vectors, coreset_weights = self.preprocess_data(
            self.full_coreset_df,
        )

        G = Coreset.coreset_to_graph(
            coreset_vectors,
            coreset_weights,
            "dist",
            self.base_config.number_of_qubits_representing_data,
        )

        bitstring = self.generate_random_bitstring(coreset_vectors)
        bitstring = "".join([str(x) for x in bitstring])

        return self.get_partition_from_bitstring(bitstring, G)

    def _get_best_bitstring(self) -> None:
        """
        Empty method to satisfy the abstract method
        """
        pass


class K3MeansClusteringKMeans(K3MeansClustering):
    def __init__(
        self,
        full_coreset_df: pd.DataFrame,
        vector_columns: list[str],
        weights_column: str,
        normalize_vectors: bool = True,
        number_of_qubits_representing_data: int = 2,
    ) -> None:
        super().__init__(
            full_coreset_df,
            vector_columns,
            weights_column,
            normalize_vectors,
            number_of_qubits_representing_data,
        )

    def run_k3_clustering(self) -> List:
        """

        Runs the K3 clustering algorithm using the KMeans approach

        Returns:
            List: The partition of the coreset into 3 clusters

        """

        coreset_vectors, _ = self.preprocess_data(
            self.full_coreset_df,
        )

        self._KMeans = KMeans(n_clusters=3).fit(coreset_vectors)

        return [
            list(
                np.where(self._KMeans.labels_ == i)[0]
                * self.base_config.number_of_qubits_representing_data
            )
            for i in range(3)
        ]

    def _get_best_bitstring(self, coreset_graph):
        pass


class K3MeansClusteringMaxCut(K3MeansClustering):
    def __init__(
        self,
        full_coreset_df: pd.DataFrame,
        vector_columns: list[str],
        weights_column: str,
        normalize_vectors: Optional[bool] = True,
        number_of_qubits_representing_data: Optional[int] = 2,
    ) -> None:
        super().__init__(
            full_coreset_df,
            vector_columns,
            weights_column,
            normalize_vectors,
            number_of_qubits_representing_data,
        )

    def run_k3_clustering(self) -> List:
        """

        Runs the K3 clustering algorithm using the MaxCut approach

        Returns:
            List: The partition of the coreset into 3 clusters

        """

        coreset_vectors, coreset_weights = self.preprocess_data(
            self.full_coreset_df,
        )

        coreset_graph = Coreset.coreset_to_graph(
            coreset_vectors,
            coreset_weights,
            "dist",
            self.base_config.number_of_qubits_representing_data,
        )

        bitstring = self._get_best_bitstring(
            coreset_graph,
        )

        return self.get_partition_from_bitstring(bitstring, coreset_graph)

    def _get_best_bitstring(self, coreset_graph: nx.Graph) -> str:
        """
        Get the best bitstring using the MaxCut approach

        Args:
            coreset_graph (nx.Graph): The coreset graph

        Returns:
            str: The best bitstring

        """

        coreset_df = self.full_coreset_df.copy()

        bitstring_length = (
            coreset_graph.number_of_nodes()
            * self.base_config.number_of_qubits_representing_data
        )

        bitstrings = self.create_all_possible_bitstrings(bitstring_length)

        lowest_cost = np.inf
        best_bitstring = None

        for bitstring in tqdm(bitstrings):
            current_bitstring_cost = 0
            partition = self.get_partition_from_bitstring(bitstring, coreset_graph)
            partition = [
                [
                    int(x / self.base_config.number_of_qubits_representing_data)
                    for x in sublist
                ]
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
