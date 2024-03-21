import random
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import cudaq
import networkx as nx
import numpy as np
import pandas as pd
from cudaq import spin
from loguru import logger
from sklearn.cluster import KMeans
from tqdm import tqdm

from ..coreset import Coreset
from ..optimizer import get_optimizer
from ..vqe_utils import get_K2_Hamiltonian, kernel_two_local
from .dendrogram import Dendrogram


class DivisiveClustering(ABC):
    @abstractmethod
    def run_divisive_clustering(self, coreset_vectors_df_for_iteration: pd.DataFrame):
        pass

    def get_hierarchical_clustering_sequence(
        self, coreset_vectors_df_for_iteration, hierarchial_sequence
    ):
        bitstring = self.run_divisive_clustering(coreset_vectors_df_for_iteration)
        return self._add_children_to_hierarchial_clustering(
            coreset_vectors_df_for_iteration, hierarchial_sequence, bitstring
        )

    def _get_iteration_coreset_vectors_and_weights(
        self, coreset_vectors_df_for_iteration
    ):
        coreset_vectors_for_iteration = coreset_vectors_df_for_iteration[
            ["X", "Y"]
        ].to_numpy()
        coreset_weights_for_iteration = coreset_vectors_df_for_iteration[
            "weights"
        ].to_numpy()

        if self._normalize_vectors:

            coreset_vectors_for_iteration = Coreset.normalize_array(
                coreset_vectors_for_iteration, True
            )
            coreset_weights_for_iteration = Coreset.normalize_array(
                coreset_weights_for_iteration
            )

        return coreset_vectors_for_iteration, coreset_weights_for_iteration

    def brute_force_cost_maxcut(self, bitstrings: list, G: nx.graph):
        """
        Cost function for brute force method

        Args:
            bitstrings: list of bit strings
            G: The graph of the problem

        Returns:
        Dictionary with bitstring and cost value
        """
        cost_value = {}
        for bitstring in tqdm(bitstrings):
            c = 0
            for i, j in G.edges():
                edge_weight = G[i][j]["weight"]
                c += self._get_edge_cost(bitstring, i, j, edge_weight)

            cost_value.update({bitstring: c})

        return cost_value

    def _get_edge_cost(self, bitstring: str, i, j, edge_weight):
        ai = int(bitstring[i])
        aj = int(bitstring[j])

        return -1 * edge_weight * (1 - ((-1) ** ai) * ((-1) ** aj))  # MaxCut equation

    def _create_all_possible_bitstrings(self, G):
        bitstrings = []
        for i in range(1, (2 ** len(G.nodes) - 1)):
            bitstrings.append(bin(i)[2:].zfill(len(G.nodes)))
        return bitstrings

    def _add_children_to_hierarchial_clustering(
        self,
        iteration_dataframe: pd.DataFrame,
        hierarchial_sequence: list,
        bitstring: str,
    ):

        iteration_dataframe["cluster"] = [int(bit) for bit in bitstring]

        for j in range(2):
            idx = list(iteration_dataframe[iteration_dataframe["cluster"] == j].index)
            if len(idx) > 0:
                hierarchial_sequence.append(idx)

        return hierarchial_sequence

    @staticmethod
    def get_divisive_cluster_cost(hierarchical_clustering_sequence, coreset_data):
        coreset_data = coreset_data.drop(["Name", "weights"], axis=1)
        cost_list = []
        for parent in hierarchical_clustering_sequence:
            children_lst = Dendrogram.find_children(
                parent, hierarchical_clustering_sequence
            )

            if not children_lst:
                continue
            else:
                children_1, children_2 = children_lst

                parent_data_frame = coreset_data.iloc[parent]

                parent_data_frame["cluster"] = 0

                parent_data_frame.loc[children_2, "cluster"] = 1

                cost = 0

                centroid_coords = parent_data_frame.groupby("cluster").mean()[
                    ["X", "Y"]
                ]
                centroid_coords = centroid_coords.to_numpy()

                for idx, row in parent_data_frame.iterrows():
                    if row.cluster == 0:
                        cost += (
                            np.linalg.norm(row[["X", "Y"]] - centroid_coords[0]) ** 2
                        )
                    else:
                        cost += (
                            np.linalg.norm(row[["X", "Y"]] - centroid_coords[1]) ** 2
                        )

                cost_list.append(cost)

        return cost_list


class DivisiveClusteringVQE(DivisiveClustering):
    def __init__(
        self,
        circuit_depth: int = 1,
        max_iterations: int = 1000,
        max_shots: int = 1000,
        threshold_for_max_cut: float = 0.5,
        normalize_vectors: bool = True,
        sort_by_descending: bool = True,
        coreset_to_graph_metric: str = "dot",
    ) -> None:
        self._circuit_depth = circuit_depth
        self._max_iterations = max_iterations
        self._max_shots = max_shots
        self._threshold_for_maxcut = threshold_for_max_cut
        self._normalize_vectors = normalize_vectors
        self._sort_by_descending = sort_by_descending
        self._coreset_to_graph_metric = coreset_to_graph_metric

    @property
    def circuit_depth(self):
        return self._circuit_depth

    @circuit_depth.setter
    def circuit_depth(self, value):
        if not isinstance(value, int) or value < 1:
            raise ValueError("circuit_depth must be an integer or greater than 0")
        self._circuit_depth = value

    @property
    def max_iterations(self):
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, value):
        if not isinstance(value, int) or value < 1:
            raise ValueError("max_iterations must be an integer or greater than 0")
        self._max_iterations = value

    @property
    def max_shots(self):
        return self._max_shots

    @max_shots.setter
    def max_shots(self, value):
        if not isinstance(value, int) or value < 1:
            raise ValueError("max_shots must be an integer or greater than 0")
        self._max_shots = value

    @property
    def threshold_for_maxcut(self):
        return self._threshold_for_maxcut

    @threshold_for_maxcut.setter
    def threshold_for_maxcut(self, value):
        if not isinstance(value, float) or value < 0 or value > 1:
            raise ValueError(
                "threshold_for_maxcut must be a float or must be between 0 and 1"
            )

    @property
    def normalize_vectors(self):
        return self._normalize_vectors

    @normalize_vectors.setter
    def normalize_vectors(self, value):
        if not isinstance(value, bool):
            raise ValueError("normalize_vectors must be a boolean")
        self._normalize_vectors = value

    @property
    def sort_by_descending(self):
        return self._sort_by_descending

    @sort_by_descending.setter
    def sort_by_descending(self, value):
        if not isinstance(value, bool):
            raise ValueError("sort_by_descending must be a boolean")
        self._sort_by_descending = value

    @property
    def coreset_to_graph_metric(self):
        return self._coreset_to_graph_metric

    @coreset_to_graph_metric.setter
    def coreset_to_graph_metric(self, value):
        if value not in ["dot", "euclidean"]:
            raise ValueError("coreset_to_graph_metric must be either dot or euclidean")
        self._coreset_to_graph_metric = value

    def run_divisive_clustering(
        self,
        coreset_vectors_df_for_iteration: pd.DataFrame,
    ):
        coreset_vectors_for_iteration_np, coreset_weights_for_iteration_np = (
            self._get_iteration_coreset_vectors_and_weights(
                coreset_vectors_df_for_iteration
            )
        )

        G = Coreset.coreset_to_graph(
            coreset_vectors_for_iteration_np,
            coreset_weights_for_iteration_np,
            metric=self._coreset_to_graph_metric,
        )

        counts = self._get_counts_from_vqe(
            G,
            self._circuit_depth,
            self._max_iterations,
            self._max_shots,
        )

        return self._get_best_bitstring(counts, G)

    def _get_counts_from_vqe(self, G, circuit_depth, max_iterations, max_shots):
        qubits = len(G.nodes)
        Hamiltonian = get_K2_Hamiltonian(G)

        kernel = kernel_two_local(qubits, circuit_depth)

        optimizer, parameter_count = get_optimizer(
            max_iterations, circuit_depth, qubits
        )

        _, optimal_parameters = cudaq.vqe(
            kernel=kernel,
            spin_operator=Hamiltonian,
            optimizer=optimizer,
            parameter_count=parameter_count,
            shots=max_shots,
        )

        counts = cudaq.sample(
            kernel,
            optimal_parameters,
            shots_count=max_shots,
        )

        return counts

    def _convert_counts_to_probability_table(self, all_bitstrings, counts):
        df = pd.DataFrame(columns=["bitstring", "probability"])
        for bitstring in all_bitstrings:
            df.loc[len(df)] = [bitstring, counts.probability(bitstring)]

        return df.sort_values("probability", ascending=self._sort_by_descending)

    def _get_best_bitstring(self, counts, G):
        all_bitstrings = self._create_all_possible_bitstrings(G)
        bitstring_probability_df = self._convert_counts_to_probability_table(
            all_bitstrings, counts
        )
        # get top percentage of the bitstring_probability_df
        if len(bitstring_probability_df) > 100:
            selected_rows = int(
                len(bitstring_probability_df) * self._threshold_for_maxcut
            )
        else:
            selected_rows = int(len(bitstring_probability_df) / 2)
        bitstring_probability_df = bitstring_probability_df.head(selected_rows)

        bitstrings = bitstring_probability_df["bitstring"].tolist()

        brute_force_cost_of_bitstrings = self.brute_force_cost_maxcut(bitstrings, G)

        return min(
            brute_force_cost_of_bitstrings, key=brute_force_cost_of_bitstrings.get
        )


class DivisiveClusteringRandom(DivisiveClustering):
    def __init__(self) -> None:
        pass

    def run_divisive_clustering(
        self,
        coreset_vectors_df_for_iteration: pd.DataFrame,
    ) -> np.ndarray:
        rows = coreset_vectors_df_for_iteration.shape[0]
        if rows > 2:
            bitstring = np.random.randint(0, 2, rows)

            bitstring_not_accepted = True

            while bitstring_not_accepted:
                if bitstring.sum() == 0 or bitstring.sum() == len(bitstring):
                    bitstring_not_accepted = True
                else:
                    bitstring_not_accepted = False
        else:
            bitstring = np.array([0, 1])

        return bitstring


class DivisiveClusteringKMeans(DivisiveClustering):
    def __init__(self) -> None:
        pass

    def run_divisive_clustering(self, coreset_vectors_df_for_iteration: pd.DataFrame):
        if len(coreset_vectors_df_for_iteration) > 2:
            coreset_vectors_df_for_iteration = coreset_vectors_df_for_iteration.drop(
                "Name", axis=1
            )
            X = coreset_vectors_df_for_iteration.to_numpy()
            kmeans = KMeans(n_clusters=2, random_state=None).fit(X)
            bitstring = kmeans.labels_

        else:
            bitstring = np.array([0, 1])

        return bitstring


class DivisiveClusteringMaxCut(DivisiveClustering):
    def __init__(
        self, normalize_vectors: bool = True, coreset_to_graph_metric: str = "dot"
    ) -> None:
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

    def run_divisive_clustering(self, coreset_vectors_df_for_iteration: pd.DataFrame):
        coreset_vectors_for_iteration_np, coreset_weights_for_iteration_np = (
            self._get_iteration_coreset_vectors_and_weights(
                coreset_vectors_df_for_iteration
            )
        )

        G = Coreset.coreset_to_graph(
            coreset_vectors_for_iteration_np,
            coreset_weights_for_iteration_np,
            metric=self._coreset_to_graph_metric,
        )

        bitstrings = self._create_all_possible_bitstrings(G)

        brute_force_bitstring_cost = self.brute_force_cost_maxcut(bitstrings, G)

        brute_force_bitstring_cost = pd.DataFrame.from_dict(
            brute_force_bitstring_cost, orient="index", columns=["cost"]
        )

        brute_force_bitstring_cost = brute_force_bitstring_cost.sort_values("cost")

        max_bitstring = brute_force_bitstring_cost.index[0]

        return [int(bit) for bit in max_bitstring]


def get_divisive_sequence(
    full_coreset_df: pd.DataFrame, divisive_clustering_function: Callable
) -> List:

    index_iteration_counter = 0
    single_clusters = 0

    index_values = list(range(len(full_coreset_df)))
    hierarchial_clustering_sequence = [index_values]

    while single_clusters < len(index_values):
        index_values_to_evaluate = hierarchial_clustering_sequence[
            index_iteration_counter
        ]
        if len(index_values_to_evaluate) == 1:
            single_clusters += 1
        else:
            coreset_vectors_df_for_iteration = full_coreset_df.iloc[
                index_values_to_evaluate
            ]

            hierarchial_clustering_sequence = (
                divisive_clustering_function.get_hierarchical_clustering_sequence(
                    coreset_vectors_df_for_iteration, hierarchial_clustering_sequence
                )
            )

        index_iteration_counter += 1

    return hierarchial_clustering_sequence
