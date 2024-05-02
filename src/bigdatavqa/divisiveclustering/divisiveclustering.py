from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

import cudaq
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

from ..coreset import Coreset
from ..plot import Dendrogram


class DivisiveClustering(ABC):
    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def run_divisive_clustering(
        self, coreset_vectors_df_for_iteration: pd.DataFrame
    ) -> Union[List[str], List[int]]:
        """
        Run the divisive clustering algorithm.

        Args:
            coreset_vectors_df_for_iteration (pd.DataFrame): The coreset vectors for the iteration.

        Returns:
            Union[List[str], List[int]]: The bitstring or the cluster. The return will depend on the name of the data point given.
        """

        pass

    def get_hierarchical_clustering_sequence(
        self,
        coreset_vectors_df_for_iteration: np.ndarray,
        hierarchial_sequence: List,
    ) -> List:
        """
        Get the hierarchical clustering sequence.

        Args:
            coreset_vectors_df_for_iteration (np.ndarray): The coreset vectors for the iteration.
            hierarchial_sequence (List): The hierarchical sequence.

        """

        bitstring = self.run_divisive_clustering(coreset_vectors_df_for_iteration)
        return self._add_children_to_hierarchial_clustering(
            coreset_vectors_df_for_iteration, hierarchial_sequence, bitstring
        )

    def _get_iteration_coreset_vectors_and_weights(
        self, coreset_vectors_df_for_iteration: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets the iteration coreset vectors and weights.

        Args:
            coreset_vectors_df_for_iteration (pd.DataFrame): The coreset vectors for the iteration.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The coreset vectors and weights.

        """

        coreset_vectors_for_iteration = coreset_vectors_df_for_iteration[["X", "Y"]].to_numpy()

        coreset_weights_for_iteration = coreset_vectors_df_for_iteration["weights"].to_numpy()

        if self.normalize_vectors:
            coreset_vectors_for_iteration = Coreset.normalize_array(
                coreset_vectors_for_iteration, True
            )
            coreset_weights_for_iteration = Coreset.normalize_array(coreset_weights_for_iteration)

        return (coreset_vectors_for_iteration, coreset_weights_for_iteration)

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

    def _get_edge_cost(self, bitstring: str, i: int, j: int, edge_weight: float) -> float:
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

        iteration_dataframe["cluster"] = [int(bit) for bit in bitstring]

        for j in range(2):
            idx = list(iteration_dataframe[iteration_dataframe["cluster"] == j].index)
            if len(idx) > 0:
                hierarchial_sequence.append(idx)

        return hierarchial_sequence

    @staticmethod
    def get_divisive_cluster_cost(
        hierarchical_clustering_sequence: List[Union[str, int]], coreset_data: pd.DataFrame
    ) -> List[float]:
        """
        Get the cost of the divisive clustering at each iteration.

        Args:
            hierarchical_clustering_sequence (List): The hierarchical clustering sequence.
            coreset_data (pd.DataFrame): The coreset data.

        Returns:
            List[float]: The cost of the divisive clustering sequence.
        """

        coreset_data = coreset_data.drop(["Name", "weights"], axis=1)
        cost_at_each_iteration = []
        for parent in hierarchical_clustering_sequence:
            children_lst = Dendrogram.find_children(parent, hierarchical_clustering_sequence)

            if not children_lst:
                continue
            else:
                _, children_2 = children_lst

                parent_data_frame = coreset_data.iloc[parent]

                parent_data_frame["cluster"] = 0

                parent_data_frame.loc[children_2, "cluster"] = 1

                cost = 0

                centroid_coords = parent_data_frame.groupby("cluster").mean()[["X", "Y"]]
                centroid_coords = centroid_coords.to_numpy()

                for idx, row in parent_data_frame.iterrows():
                    if row.cluster == 0:
                        cost += np.linalg.norm(row[["X", "Y"]] - centroid_coords[0]) ** 2
                    else:
                        cost += np.linalg.norm(row[["X", "Y"]] - centroid_coords[1]) ** 2

                cost_at_each_iteration.append(cost)

        return cost_at_each_iteration

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
            selected_rows = int(len(bitstring_probability_df) * self.threshold_for_maxcut)
        else:
            selected_rows = int(len(bitstring_probability_df) / 2)

        bitstring_probability_df = bitstring_probability_df.head(selected_rows)

        bitstrings = bitstring_probability_df["bitstring"].tolist()

        brute_force_cost_of_bitstrings = self.brute_force_cost_maxcut(bitstrings, G)

        return min(brute_force_cost_of_bitstrings, key=brute_force_cost_of_bitstrings.get)

    def get_divisive_sequence(self, full_coreset_df: pd.DataFrame) -> List[Union[str, int]]:
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
        hierarchial_clustering_sequence = [index_values]

        while single_clusters < len(index_values):
            index_values_to_evaluate = hierarchial_clustering_sequence[index_iteration_counter]
            if len(index_values_to_evaluate) == 1:
                single_clusters += 1

            elif len(index_values_to_evaluate) == 2:
                hierarchial_clustering_sequence.append([index_values_to_evaluate[0]])
                hierarchial_clustering_sequence.append([index_values_to_evaluate[1]])

            else:
                coreset_vectors_df_for_iteration = full_coreset_df.iloc[index_values_to_evaluate]

                hierarchial_clustering_sequence = self.get_hierarchical_clustering_sequence(
                    coreset_vectors_df_for_iteration,
                    hierarchial_clustering_sequence,
                )

            index_iteration_counter += 1

        return hierarchial_clustering_sequence


class DivisiveClusteringVQA(DivisiveClustering):
    def __init__(
        self,
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

    def run_divisive_clustering(
        self,
        coreset_vectors_df_for_iteration: pd.DataFrame,
    ):
        coreset_vectors_for_iteration_np, coreset_weights_for_iteration_np = (
            self._get_iteration_coreset_vectors_and_weights(coreset_vectors_df_for_iteration)
        )

        G = Coreset.coreset_to_graph(
            coreset_vectors_for_iteration_np,
            coreset_weights_for_iteration_np,
            metric=self.coreset_to_graph_metric,
        )

        counts = self.get_counts_from_simulation(
            G,
            self.circuit_depth,
            self.max_iterations,
            self.max_shots,
        )

        return self._get_best_bitstring(counts, G)

    def get_counts_from_simulation(self, G, circuit_depth, max_iterations, max_shots):
        qubits = len(G.nodes)
        Hamiltonian = self.create_Hamiltonian(G)
        optimizer, parameter_count = self.optimizer_function(
            self.optimizer, max_iterations, qubits=qubits, circuit_depth=circuit_depth
        )

        kernel = self.create_circuit(qubits, circuit_depth)

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

        return counts


class DivisiveClusteringRandom(DivisiveClustering):
    def __init__(self) -> None:
        pass

    def run_divisive_clustering(
        self,
        coreset_vectors_df_for_iteration: pd.DataFrame,
    ) -> List[int]:
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
            X = coreset_vectors_df_for_iteration[["X", "Y"]].to_numpy()
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

    def _create_all_possible_bitstrings(self, G):
        bitstrings = []
        for i in range(1, (2 ** len(G.nodes) - 1)):
            bitstrings.append(bin(i)[2:].zfill(len(G.nodes)))
        return bitstrings

    def run_divisive_clustering(self, coreset_vectors_df_for_iteration: pd.DataFrame):
        coreset_vectors_for_iteration_np, coreset_weights_for_iteration_np = (
            self._get_iteration_coreset_vectors_and_weights(coreset_vectors_df_for_iteration)
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

        best_bitstring = brute_force_bitstring_cost["cost"].idxmax()

        return [int(bit) for bit in best_bitstring]
