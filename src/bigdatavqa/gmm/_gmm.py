from abc import abstractmethod
from typing import Callable, List, Optional, Union

from .._base import BigDataVQA

import cudaq
import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from numpy.linalg import inv
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


class GMMClustering(BigDataVQA):
    def __init__(
        self,
        full_coreset_df: pd.DataFrame,
        vector_columns: List[str],
        weights_column: str,
        normalize_vectors: Optional[bool] = True,
    ) -> None:
        """

        Parameters
        ----------
        full_coreset_df : pd.DataFrame
            The dataframe containing the coreset data.
        vector_columns : list
            The columns containing the vector data.
        weights_column : str
            The column containing the weights of the vectors.
        normalize_vectors : bool, optional
            Whether to normalize the vectors, by default True.

        """
        super().__init__(
            full_coreset_df=full_coreset_df,
            vector_columns=vector_columns,
            weights_column=weights_column,
            mormalize_vectors=normalize_vectors,
            number_of_qubits_representing_data=1,
        )

        self.cost = np.inf

    def fit(self) -> None:
        """

        Fit the GMM model to the data.

        """
        self.labels = self.run_GMM()
        self.cluster_centers = self.get_cluster_centroids_from_bitstring()
        if self.cost == np.inf:
            self.cost = self.get_cost_using_kmeans_approach(
                self.full_coreset_df, self.cluster_centers
            )

    @abstractmethod
    def run_GMM(
        self,
        *args,
        **kwargs,
    ):
        """

        Run the GMM algorithm. This method should be implemented in the child classes and it will vary depending on the GMM algorithm used.

        """
        pass

    def get_cluster_centroids_from_bitstring(
        self, bitstring: Optional[str] = None
    ) -> np.ndarray:
        """

        Get the cluster centroids from the bitstring.

        Args:
        bitstring (Optional[str], optional): The bitstring. Defaults to None.

        Returns:
        np.ndarray: The cluster centroids.

        """

        columns_retain = self.vector_columns + ["label"]

        if bitstring is None:
            self.full_coreset_df["label"] = self.labels
        else:
            self.full_coreset_df["label"] = [int(i) for i in bitstring]

        return self.full_coreset_df[columns_retain].groupby("label").mean().values

    def plot(self) -> None:
        """

        Plot the clustering outcome over the coreset data

        """
        plt.scatter(
            self.full_coreset_df["X"],
            self.full_coreset_df["Y"],
            c=self.full_coreset_df["label"],
            label="coreset vectors",
            cmap="viridis",
        )

        plt.scatter(
            self.cluster_centers[:, 0],
            self.cluster_centers[:, 1],
            marker="*",
            color="r",
            label="Centers",
        )
        plt.legend()
        plt.show()


class GMMClusteringVQA(GMMClustering):
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
        create_Hamiltonian: Callable,
        max_iterations: int,
        max_shots: int,
        normalize_vectors: Optional[bool] = True,
    ) -> None:
        super().__init__(
            full_coreset_df,
            vector_columns,
            weights_column,
            normalize_vectors,
        )
        self.qubits = qubits
        self.create_circuit = create_circuit
        self.circuit_depth = circuit_depth
        self.optimizer_function = optimizer_function
        self.optimizer = optimizer
        self.create_Hamiltonian = create_Hamiltonian
        self.max_iterations = max_iterations
        self.max_shots = max_shots

    def Z_i(self, i: int, length: int) -> str:
        """
        if index i is in the range 0, ..., length-1, the function returns the operator Z_i
        else: the funtion returns the pauli string consisting of pauli I's only
        length is the number of pauli operators tensorised

        Args:
        i (int): The index of the Z operator.
        length (int): The length of the pauli string.

        Returns:
        str: The pauli string.
        """
        pauli_string = ""
        for j in range(length):
            if i == j:
                pauli_string += "Z"
            else:
                pauli_string += "I"
        return pauli_string

    def Z_ij(self, i: int, j: int, length: int) -> str:
        """
        if index i and j are in the range 0, ..., length-1, the function returns the operator Z_iZ_j
        else: the funtion returns the pauli string consisting of pauli I's only
        length is the number of pauli operators tensorised

        Args:
            i (int): The index of the Z operator.
            j (int): The index of the Z operator.
            length (int): The length of the pauli string.

        Returns:
            str: The pauli string.

        """
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

    def get_scatter_matrix(
        self, coreset_vectors: np.ndarray, coreset_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get the scatter matrix of the coreset data.

        Args:
            coreset_vectors (np.ndarray): The coreset vectors.
            coreset_weights (Optional[np.ndarray], optional): The coreset weights. Defaults to None.

        Returns:
            np.ndarray: The scatter matrix.

        """

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

    def create_pauli_operators(
        self, coreset_vectors: np.ndarray, coreset_weights: np.ndarray
    ) -> List[List[Union[str, float]]]:
        """

        Create the pauli operators.

        Args:
            coreset_vectors (np.ndarray): The coreset vectors.
            coreset_weights (np.ndarray): The coreset weights.

        Returns:
            List[List[str, float]]: The pauli operators.

        """

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

    def run_GMM(
        self,
        *args,
        **kwargs,
    ) -> List[int]:
        """

        Run the GMM algorithm using the VQA approach.

        Returns:
            List[int]: The best bitstring.

        """
        coreset_vectors, coreset_weights = self.preprocess_data(
            self.full_coreset_df,
            self.vector_columns,
            self.weights_column,
            self.normalize_vectors,
        )

        optimizer, parameter_count = self.optimizer_function(
            self.optimizer,
            self.max_iterations,
            qubits=self.qubits,
            circuit_depth=self.circuit_depth,
        )

        pauli_operators = self.create_pauli_operators(coreset_vectors, coreset_weights)

        kernel = self.create_circuit(self.qubits, self.circuit_depth)

        Hamiltonian = self.create_Hamiltonian(pauli_operators)

        counts = self.get_counts(
            self.qubits, Hamiltonian, kernel, optimizer, parameter_count
        )

        return self._get_best_bitstring(counts)

    def _get_best_bitstring(self, counts: cudaq.SampleResult) -> List[int]:
        best_bitstring = counts.most_probable()
        return [int(i) for i in best_bitstring]


class GMMClusteringRandom(GMMClustering):
    def __init__(
        self,
        full_coreset_df: pd.DataFrame,
        vector_columns: List[str],
        weights_column: str,
        normalize_vectors=True,
    ):
        super().__init__(
            full_coreset_df, vector_columns, weights_column, normalize_vectors
        )

    def run_GMM(
        self,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """

        Run the GMM algorithm using the random approach.

        Returns:
            np.ndarray: The best bitstring.

        """

        coreset_vectors, _ = self.preprocess_data(
            self.full_coreset_df,
            self.vector_columns,
            self.weights_column,
            self.normalize_vectors,
        )
        return self.generate_random_bitstring(coreset_vectors)

    def _get_best_bitstring(self, *args, **kwargs) -> None:
        """

        Empty method to be implemented in the child classes to satisfy the abstract method.

        """
        pass


class GMMClusteringClassicalGMM(GMMClustering):
    def __init__(
        self,
        full_coreset_df: pd.DataFrame,
        vector_columns: List[str],
        weights_column: str,
        normalize_vectors: Optional[bool] = True,
    ):
        super().__init__(
            full_coreset_df, vector_columns, weights_column, normalize_vectors
        )

    def run_GMM(
        self,
        n_components: Optional[int] = 2,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """
        Run the GMM algorithm using the classical GMM approach.

        Returns:
            np.ndarray: The best bitstring.

        """
        coreset_vectors, _ = self.preprocess_data(
            self.full_coreset_df,
            self.vector_columns,
            self.weights_column,
            self.normalize_vectors,
        )
        gmm = GaussianMixture(n_components=n_components)
        return self._get_best_bitstring(gmm, coreset_vectors)

    def _get_best_bitstring(
        self, gmm_object: sklearn.mixture.GaussianMixture, coreset_vectors: np.ndarray
    ) -> np.ndarray:
        return gmm_object.fit_predict(coreset_vectors)


class GMMClusteringMaxCut(GMMClustering):
    def __init__(
        self,
        full_coreset_df: pd.DataFrame,
        vector_columns: List[str],
        weights_column: str,
        normalize_vectors: Optional[bool] = True,
    ):
        super().__init__(
            full_coreset_df, vector_columns, weights_column, normalize_vectors
        )

    def run_GMM(
        self,
        *args,
        **kwargs,
    ) -> str:
        """

        Run the GMM algorithm using the MaxCut approach.

        Returns:
            str: The best bitstring.
        """
        coreset_vectors, _ = self.preprocess_data(
            self.full_coreset_df,
            self.vector_columns,
            self.weights_column,
            self.normalize_vectors,
        )
        bitstring_length = len(coreset_vectors)
        bitstrings = self.create_all_possible_bitstrings(bitstring_length)

        lowest_cost = np.inf
        best_bitstring = None

        for bitstring in tqdm(bitstrings):
            cluster_centers = self.get_cluster_centroids_from_bitstring(bitstring)
            current_bitstring_cost = self.get_cost_using_kmeans_approach(
                self.full_coreset_df, cluster_centers
            )

            if current_bitstring_cost < self.cost:
                best_bitstring = bitstring
                self.cost = current_bitstring_cost
                self.cluster_centers = cluster_centers

        return best_bitstring

    def _get_best_bitstring(self, *args, **kwargs):
        """

        Empty method to be implemented in the child classes to satisfy the abstract method.

        """
        pass
