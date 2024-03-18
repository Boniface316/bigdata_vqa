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
from ..postexecution import add_children_to_hierachial_clustering, get_best_bitstring
from ..vqe_utils import get_K2_Hamiltonian, kernel_two_local


class DivisiveClustering(ABC):
    @abstractmethod
    def run_divisive_clustering(self, coreset_vectors_df_for_iteration: pd.DataFrame):
        pass

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


def get_divisive_sequence(
    full_coreset_df: pd.DataFrame, divisive_clustering_function: Callable
) -> List:
    """
    Creates a hierarchical cluster using divisive clustering algorithm.

    Args:
        raw_data (np.ndarray): The input data for clustering.
        number_of_qubits (int): The number of qubits to be used in the clustering.
        number_of_centroids_evaluation (int): The number of centroids to evaluate when creating the coreset.
        number_of_coresets_to_evaluate (int): The number of coreset vectors to evaluate when creating the coreset.
        max_shots (int, optional): The maximum number of shots for quantum measurements. Defaults to 1000.
        max_iterations (int, optional): The maximum number of iterations for the clustering algorithm. Defaults to 1000.
        circuit_depth (int, optional): The depth of the quantum circuit. Defaults to 1.

    Returns:
        Tuple[List[int], List[np.ndarray]]: A tuple containing the hierarchical clustering sequence and the coreset vectors and weights.

    """

    index_iteration_counter = 0
    single_clusters = 0

    index_values = list(range(len(full_coreset_df)))
    hierarchial_clustering_sequence = [index_values]

    # variables = ["full_coreset_pd"]

    # if clustering_method == "vqe":
    #     divisive_clustering_function = _divisive_clustering_using_vqe
    #     variables.extend(["circuit_depth", "max_iterations", "max_shots"])
    # elif clustering_method == "random":
    #     divisive_clustering_function = _divisive_clustering_using_random
    # elif clustering_method == "kmeans":
    #     divisive_clustering_function = _divisive_clustering_using_kmeans
    # elif clustering_method == "maxcut":
    #     divisive_clustering_function = _divisive_clustering_using_maxcut
    # else:
    #     raise ValueError("Method not found")

    # self._check_variables(variables, clustering_method)

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

            bitstring = divisive_clustering_function.run_divisive_clustering(
                coreset_vectors_df_for_iteration
            )

            print(bitstring)

            hierarchial_clustering_sequence = add_children_to_hierachial_clustering(
                coreset_vectors_df_for_iteration,
                hierarchial_clustering_sequence,
                bitstring,
            )

        index_iteration_counter += 1

    return hierarchial_clustering_sequence


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
            all_bitstrings, counts, self._sort_by_descending
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
                "name", axis=1
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
