from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

from .._base import BigDataVQA
from ..coreset import Coreset
from ..plot import Dendrogram, Voironi_Tessalation

import cudaq
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm


class DivisiveClustering(BigDataVQA, Dendrogram, Voironi_Tessalation):
    def __init__(
        self,
        full_coreset_df: pd.DataFrame,
        vector_columns: List[str],
        weight_column: str,
    ) -> None:
        self.full_coreset_df = full_coreset_df
        self.vector_columns = vector_columns
        self.weight_column = weight_column
        self.linkage_matrix = []

    @property
    def coreset_data(self) -> pd.DataFrame:
        return self.coreset_data

    @coreset_data.setter
    def coreset_data(self, coreset_data: pd.DataFrame) -> None:
        self.linkage_matrix = []
        self.coreset_data = coreset_data

    @property
    def hierarchical_clustering_sequence(self) -> List[Union[str, int]]:
        return self._hierarchical_clustering_sequence

    @hierarchical_clustering_sequence.setter
    def hierarchical_clustering_sequence(
        self, hierarchical_clustering_sequence: List[Union[str, int]]
    ) -> None:
        self.linkage_matrix = []
        self._hierarchical_clustering_sequence = hierarchical_clustering_sequence

    def fit(self):
        self.hierarchical_clustering_sequence = self._divisive_sequence(
            self.full_coreset_df, self.vector_columns, self.weight_column
        )

        self._get_linkage_matrix(self.hierarchical_clustering_sequence[0])

        self.cost_at_iterations = self.get_divisive_cluster_cost()

        self.cost = sum(self.cost_at_iterations)

    def _divisive_sequence(
        self,
        full_coreset_df: pd.DataFrame,
        vector_columns: List[str],
        weight_column: str = "weights",
    ) -> List[Union[str, int]]:
        """
        Perform divisive clustering on the coreset data.
        Args:
            full_coreset_df (pd.DataFrame): The full coreset data.
        Returns:
            List[Union[str, int]]: The hierarchical clustering sequence.
        """

        index_iteration_counter = 0
        single_clusters = 0

        index_values = list(range(len(full_coreset_df)))
        hierarchical_clustering_sequence = [index_values]

        while single_clusters < len(index_values):
            index_values_to_evaluate = hierarchical_clustering_sequence[
                index_iteration_counter
            ]
            if len(index_values_to_evaluate) == 1:
                single_clusters += 1

            elif len(index_values_to_evaluate) == 2:
                hierarchical_clustering_sequence.append([index_values_to_evaluate[0]])
                hierarchical_clustering_sequence.append([index_values_to_evaluate[1]])

            else:
                coreset_vectors_df_for_iteration = full_coreset_df.iloc[
                    index_values_to_evaluate
                ]

                hierarchical_clustering_sequence = (
                    self._get_hierarchical_clustering_sequence(
                        coreset_vectors_df_for_iteration,
                        hierarchical_clustering_sequence,
                        vector_columns,
                        weight_column,
                    )
                )

            index_iteration_counter += 1

        return hierarchical_clustering_sequence

    def _get_hierarchical_clustering_sequence(
        self,
        coreset_vectors_df_for_iteration: np.ndarray,
        hierarchial_sequence: List,
        vector_columns: List[str],
        weight_column: str = "weights",
    ) -> List:
        """
        Get the hierarchical clustering sequence.
        Args:
            coreset_vectors_df_for_iteration (np.ndarray): The coreset vectors for the iteration.
            hierarchial_sequence (List): The hierarchical sequence.
        """

        bitstring = self._run_divisive_clustering(
            coreset_vectors_df_for_iteration, vector_columns, weight_column
        )
        return self._add_children_to_hierarchial_clustering(
            coreset_vectors_df_for_iteration, hierarchial_sequence, bitstring
        )

    @abstractmethod
    def _run_divisive_clustering(
        self,
        coreset_vectors_df_for_iteration: pd.DataFrame,
        vector_columns: List[str],
        weights_column: str,
    ) -> Union[List[str], List[int]]:
        """
        Run the divisive clustering algorithm.
        Args:
            coreset_vectors_df_for_iteration (pd.DataFrame): The coreset vectors for the iteration.
        Returns:
            Union[List[str], List[int]]: The bitstring or the cluster. The return will depend on the name of the data point given.
        """

        pass

    def _add_children_to_hierarchial_clustering(
        self,
        iteration_dataframe: pd.DataFrame,
        hierarchial_sequence: list,
        bitstring: str,
    ) -> List[Union[str, int]]:
        """
        Add children to the hierarchical clustering sequence.
        Args:
            iteration_dataframe (pd.DataFrame): The iteration dataframe.
            hierarchial_sequence (list): The hierarchical sequence.
            bitstring (str): The bitstring.
        Returns:
            list: The hierarchical sequence.
        """

        iteration_dataframe["label"] = [int(bit) for bit in bitstring]

        for j in range(2):
            idx = list(iteration_dataframe[iteration_dataframe["label"] == j].index)
            if len(idx) > 0:
                hierarchial_sequence.append(idx)

        return hierarchial_sequence

    def get_divisive_cluster_cost(self) -> List[float]:
        """
        Get the cost of the divisive clustering at each iteration.
        Args:
            hierarchical_clustering_sequence (List): The hierarchical clustering sequence.
            coreset_data (pd.DataFrame): The coreset data.
        Returns:
            List[float]: The cost of the divisive clustering sequence.
        """

        coreset_data = self.full_coreset_df[self.vector_columns]
        cost_at_each_iteration = []
        for parent in self.hierarchical_clustering_sequence:
            children_lst = Dendrogram.find_children(
                parent, self.hierarchical_clustering_sequence
            )

            if not children_lst:
                continue
            else:
                parent_data_frame = self._get_parent_data_frame(
                    parent, children_lst, coreset_data
                )

                centroid_coords = parent_data_frame.groupby("label").mean()[
                    self.vector_columns
                ]
                centroid_coords = centroid_coords.to_numpy()

                cost = super().get_cost_using_kmeans_approach(
                    parent_data_frame, centroid_coords
                )

                cost_at_each_iteration.append(cost)

        return cost_at_each_iteration

    def _get_parent_data_frame(self, parent, children_lst, coreset_data):
        _, children_2 = children_lst

        parent_data_frame = coreset_data.iloc[parent]

        parent_data_frame["label"] = 0

        parent_data_frame.loc[children_2, "label"] = 1

        return parent_data_frame

    def _get_best_bitstring(self, counts: cudaq.SampleResult, G: nx.Graph) -> str:
        """
        From the simulator output, extract the best bitstring.
        Args:
            counts (cudaq.SampleResult): The counts.
            G (nx.Graph): The graph.
        Returns:
            str: The best bitstring.
        """

        counts_pd = pd.DataFrame(counts.items(), columns=["bitstring", "counts"])
        counts_pd["probability"] = counts_pd["counts"] / counts_pd["counts"].sum()
        bitstring_probability_df = counts_pd.drop(columns=["counts"])
        bitstring_probability_df = bitstring_probability_df.sort_values(
            "probability", ascending=self.sort_by_descending
        )

        unacceptable_bitstrings = [
            "".join("1" for _ in range(10)),
            "".join("0" for _ in range(10)),
        ]

        bitstring_probability_df = bitstring_probability_df[
            ~bitstring_probability_df["bitstring"].isin(unacceptable_bitstrings)
        ]

        if len(bitstring_probability_df) > 10:
            selected_rows = int(
                len(bitstring_probability_df) * self.threshold_for_maxcut
            )
        else:
            selected_rows = int(len(bitstring_probability_df) / 2)

        bitstring_probability_df = bitstring_probability_df.head(selected_rows)

        bitstrings = bitstring_probability_df["bitstring"].tolist()

        brute_force_cost_of_bitstrings = self.brute_force_cost_maxcut(bitstrings, G)

        return min(
            brute_force_cost_of_bitstrings, key=brute_force_cost_of_bitstrings.get
        )

    def brute_force_cost_maxcut(
        self, bitstrings: list[Union[str, int]], G: nx.graph
    ) -> Dict[str, float]:
        """
        Cost function for brute force method
        Args:
            bitstrings: list of bit strings
            G: The graph of the problem
        Returns:
            Dict: Dictionary with bitstring and cost value
        """

        cost_value = {}
        for bitstring in tqdm(bitstrings):
            edge_cost = 0
            for edge_i, edge_j in G.edges():
                edge_weight = G[edge_i][edge_j]["weight"]
                edge_cost += self._get_edge_cost(bitstring, edge_i, edge_j, edge_weight)

            cost_value.update({bitstring: edge_cost})

        return cost_value

    def _get_edge_cost(
        self, bitstring: str, i: int, j: int, edge_weight: float
    ) -> float:
        """
        Get the edge cost using MaxCut cost function.
        Args:
            bitstring: The bitstring
            i: The first node
            j: The second node
            edge_weight: The edge weight
        Returns:
            float: The edge cost
        """

        ai = int(bitstring[i])
        aj = int(bitstring[j])

        return -1 * edge_weight * (1 - ((-1) ** ai) * ((-1) ** aj))


class DivisiveClusteringVQA(DivisiveClustering):
    def __init__(
        self,
        coreset_df: pd.DataFrame,
        vector_columns: List[str],
        weights_column: str,
        circuit_depth: int,
        max_iterations: int,
        max_shots: int,
        threshold_for_max_cut: float,
        create_Hamiltonian: Callable,
        optimizer: cudaq.optimizers.optimizer,
        optimizer_function: Callable,
        create_circuit: Callable,
        normalize_vectors: Optional[bool] = True,
        sort_by_descending: Optional[bool] = True,
        coreset_to_graph_metric: Optional[str] = "dot",
    ) -> None:
        super().__init__(coreset_df, vector_columns, weights_column)
        self.circuit_depth = circuit_depth
        self.max_iterations = max_iterations
        self.max_shots = max_shots
        self.threshold_for_maxcut = threshold_for_max_cut
        self.normalize_vectors = normalize_vectors
        self.sort_by_descending = sort_by_descending
        self.coreset_to_graph_metric = coreset_to_graph_metric
        self.create_Hamiltonian = create_Hamiltonian
        self.create_circuit = create_circuit
        self.optimizer = optimizer
        self.optimizer_function = optimizer_function

    def _run_divisive_clustering(
        self,
        coreset_vectors_df_for_iteration: pd.DataFrame,
        vector_columns: List[str],
        weights_column: str = "weights",
    ):
        coreset_vectors_for_iteration_np, coreset_weights_for_iteration_np = (
            self.preprocess_data(
                coreset_vectors_df_for_iteration,
                vector_columns,
                weights_column,
            )
        )

        G = Coreset.coreset_to_graph(
            coreset_vectors_for_iteration_np,
            coreset_weights_for_iteration_np,
            metric=self.coreset_to_graph_metric,
        )

        Hamiltonian = self.create_Hamiltonian(G)
        optimizer, parameter_count = self.optimizer_function(
            self.optimizer,
            self.max_iterations,
            qubits=len(G.nodes),
            circuit_depth=self.circuit_depth,
        )

        kernel = self.create_circuit(len(G.nodes), self.circuit_depth)

        counts = self.get_counts(
            len(G.nodes),
            Hamiltonian,
            kernel,
            optimizer,
            parameter_count,
        )

        return self._get_best_bitstring(counts, G)


class DivisiveClusteringRandom(DivisiveClustering):
    def __init__(self, coreset_df, vector_columns, weights_column) -> None:
        super().__init__(coreset_df, vector_columns, weights_column)

    def _run_divisive_clustering(
        self,
        coreset_vectors_df_for_iteration: pd.DataFrame,
        *args,
        **kwargs,
    ) -> List[int]:
        return self.generate_random_bitstring(coreset_vectors_df_for_iteration)


class DivisiveClusteringKMeans(DivisiveClustering):
    def __init__(self, coreset_df, vector_columns, weights_column) -> None:
        super().__init__(coreset_df, vector_columns, weights_column)

    def _run_divisive_clustering(
        self,
        coreset_vectors_df_for_iteration: pd.DataFrame,
        vector_columns,
        *args,
        **kwargs,
    ):
        if len(coreset_vectors_df_for_iteration) > 2:
            X = coreset_vectors_df_for_iteration[vector_columns].to_numpy()
            kmeans = KMeans(n_clusters=2, random_state=None).fit(X)
            bitstring = kmeans.labels_

        else:
            bitstring = np.array([0, 1])

        return bitstring


class DivisiveClusteringMaxCut(DivisiveClustering):
    def __init__(
        self,
        coreset_df,
        vector_columns,
        weights_column,
        normalize_vectors: bool = True,
        coreset_to_graph_metric: str = "dot",
    ) -> None:
        super().__init__(coreset_df, vector_columns, weights_column)
        self._normalize_vectors = normalize_vectors
        self._coreset_to_graph_metric = coreset_to_graph_metric

    @property
    def normalize_vectors(self):
        return self._normalize_vectors

    @normalize_vectors.setter
    def normalize_vectors(self, value):
        if not isinstance(value, bool):
            raise ValueError("normalize_vectors must be a boolean")
        self._normalize_vectors = value

    def _run_divisive_clustering(
        self,
        coreset_vectors_df_for_iteration: pd.DataFrame,
        vector_columns: List[str],
        weights_column: str = "weights",
    ):
        coreset_vectors_for_iteration_np, coreset_weights_for_iteration_np = (
            self.preprocess_data(
                coreset_vectors_df_for_iteration, vector_columns, weights_column
            )
        )

        G = Coreset.coreset_to_graph(
            coreset_vectors_for_iteration_np,
            coreset_weights_for_iteration_np,
            metric=self._coreset_to_graph_metric,
        )

        bitstrings = self._create_all_possible_bitstrings(len(G.nodes))

        brute_force_bitstring_cost = self.brute_force_cost_maxcut(bitstrings, G)

        brute_force_bitstring_cost = pd.DataFrame.from_dict(
            brute_force_bitstring_cost, orient="index", columns=["cost"]
        )

        brute_force_bitstring_cost = brute_force_bitstring_cost.sort_values("cost")

        best_bitstring = brute_force_bitstring_cost["cost"].idxmax()

        return [int(bit) for bit in best_bitstring]
