from abc import ABC, abstractmethod
from typing import Optional

from bigdatavqa.coreset import Coreset

import cudaq
import numpy as np
import pandas as pd


class BigDataVQA(ABC):
    def __init__(
        self,
        full_coreset_df: pd.DataFrame,
        vector_columns: list[str],
        weights_column: str,
        mormalize_vectors: bool,
        number_of_qubits_representing_data: int,
    ) -> None:
        """
        Initialize the BigDataVQA class.

        Args:
            full_coreset_df (pd.DataFrame): The full coreset DataFrame.
            vector_columns (list[str]): The columns that represent the vectors.
            weights_column (str): The column that represents the weights.
            mormalize_vectors (bool): Whether to normalize the vectors.
            number_of_qubits_representing_data (int): The number of qubits representing the data.
        """
        self.full_coreset_df = full_coreset_df
        self.vector_columns = vector_columns
        self.weights_column = weights_column
        self.normalize_vectors = mormalize_vectors
        self.number_of_qubits_representing_data = number_of_qubits_representing_data

    def preprocess_data(
        self,
        coreset_df: pd.DataFrame,
        vector_columns: list[str],
        weight_columns: str,
        normalize_vectors: Optional[bool] = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data.
        Args:
            coreset_df (pd.DataFrame): The coreset DataFrame.
            vector_columns (list[str]): The columns that represent the vectors.
            weight_columns (str): The column that represents the weights.
            normalize_vectors (bool, optional): Whether to normalize the vectors. Defaults to True.
        """
        coreset_vectors = coreset_df[vector_columns].to_numpy()
        coreset_weights = coreset_df[weight_columns].to_numpy()

        if normalize_vectors:
            coreset_vectors = Coreset.normalize_array(coreset_vectors, True)
            coreset_weights = Coreset.normalize_array(coreset_weights)

        return coreset_vectors, coreset_weights

    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        """
        Runs the algorithm to fit the data. This will vary depending on the algorithm.

        """
        pass

    @abstractmethod
    def _get_best_bitstring(self, *args, **kwargs) -> str:
        """
        Get the best bitstring. This will vary depending on the algorithm.

        Returns:
            str: The best bitstring.

        """
        pass

    def generate_random_bitstring(self, coreset_vectors: np.ndarray) -> np.ndarray:
        """
        Generate a random bitstring.

        Args:
            coreset_vectors (np.ndarray): The coreset vectors.

        Returns:
            np.ndarray: The generated bitstring.
        """

        bitstring_length = len(coreset_vectors)
        if bitstring_length > 2:
            bitstring_not_accepted = True

            while bitstring_not_accepted:
                bitstring = np.random.randint(0, 2, bitstring_length)
                if bitstring.sum() == 0 or bitstring.sum() == len(bitstring):
                    bitstring_not_accepted = True
                else:
                    bitstring_not_accepted = False
        else:
            bitstring = np.array([0, 1])

        return bitstring

    def get_cost_using_kmeans_approach(
        self, coreset_df: pd.DataFrame, cluster_centers: np.array, *args, **kwargs
    ) -> float:
        """
        Calculate the cost using the method used in KMeans. This is the sum of the squared distances of each point to its cluster center.

        Args:
            coreset_df (pd.DataFrame): The coreset DataFrame.
            cluster_centers (np.array): The cluster centers.

        Returns:
            float: The cost.

        """
        cumulative_cost = 0
        for label, grouped_by_label in coreset_df.groupby("label"):
            cluster_center = cluster_centers[label]
            for _, row in grouped_by_label.iterrows():
                cumulative_cost += (
                    np.linalg.norm(row[self.vector_columns] - cluster_center) ** 2
                )

        return cumulative_cost

    def create_all_possible_bitstrings(self, bitstring_length: int) -> list[str]:
        """
        Create all possible bitstrings.

        Args:
            bitstring_length (int): The length of the bitstring.

        Returns:
            list[str]: The list of all possible bitstrings.

        """

        return [
            format(i, f"0{bitstring_length}b")
            for i in range(1, (2**bitstring_length) - 1)
        ]

    def get_counts(
        self,
        qubits: int,
        Hamiltonian: cudaq.SpinOperator,
        kernel: cudaq.Kernel,
        optimizer: cudaq.optimizers.optimizer,
        parameter_count: int,
    ) -> cudaq.SampleResult:
        """
        Get shot counts from the circuit execution.

        Args:
            qubits (int): The number of qubits.
            Hamiltonian (cudaq.SpinOperator): The Hamiltonian.
            kernel (cudaq.Kernel): The kernel.
            optimizer (cudaq.optimizers.optimizer): The optimizer.
            parameter_count (int): The parameter count.
        """

        def objective_function(
            parameter_vector: list[float],
            Hamiltonian=Hamiltonian,
            kernel=kernel,
        ) -> tuple[float, list[float]]:
            get_result = lambda parameter_vector: cudaq.observe(
                kernel, Hamiltonian, parameter_vector, qubits, self.circuit_depth
            ).expectation()

            return get_result(parameter_vector)

        self.energy, self.optimal_parameters = optimizer.optimize(
            dimensions=parameter_count, function=objective_function
        )

        return cudaq.sample(
            kernel,
            self.optimal_parameters,
            qubits,
            self.circuit_depth,
            shots_count=self.max_shots,
        )
