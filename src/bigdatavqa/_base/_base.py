from abc import ABC, abstractmethod

from bigdatavqa.coreset import Coreset

import cudaq
import numpy as np


class BigDataVQA(ABC):
    def __init__(self, full_coreset_df, vector_columns, weights_column):
        self.full_coreset_df = full_coreset_df
        self.vector_columns = vector_columns
        self.weights_column = weights_column

    def preprocess_data(
        self, coreset_df, vector_columns, weight_columns, normalize_vectors=True
    ):
        coreset_vectors = coreset_df[vector_columns].to_numpy()
        coreset_weights = coreset_df[weight_columns].to_numpy()

        if normalize_vectors:
            coreset_vectors = Coreset.normalize_array(coreset_vectors, True)
            coreset_weights = Coreset.normalize_array(coreset_weights)

        return coreset_vectors, coreset_weights

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def _get_best_bitstring(self, *args, **kwargs):
        pass

    def generate_random_bitstring(self, coreset_vectors):
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
        self, coreset_df, cluster_centers, *args, **kwargs
    ):
        cumulative_cost = 0
        for label, grouped_by_label in coreset_df.groupby("label"):
            cluster_center = cluster_centers[label]
            for _, row in grouped_by_label.iterrows():
                cumulative_cost += (
                    np.linalg.norm(row[self.vector_columns] - cluster_center) ** 2
                )

        return cumulative_cost

    def create_all_possible_bitstrings(self, bitstring_length):
        return [
            format(i, f"0{bitstring_length}b")
            for i in range(1, (2**bitstring_length) - 1)
        ]

    def get_counts(self, qubits, Hamiltonian, kernel, optimizer, parameter_count):
        def objective_function(
            parameter_vector: list[float],
            Hamiltonian=Hamiltonian,
            kernel=kernel,
        ) -> tuple[float, list[float]]:
            get_result = lambda parameter_vector: cudaq.observe(
                kernel, Hamiltonian, parameter_vector, qubits, self.circuit_depth
            ).expectation()

            return get_result(parameter_vector)

        energy, optimal_parameters = optimizer.optimize(
            dimensions=parameter_count, function=objective_function
        )

        return cudaq.sample(
            kernel,
            optimal_parameters,
            qubits,
            self.circuit_depth,
            shots_count=self.max_shots,
        )

    def get_Hamiltonian(self, G):
        qubits = len(G.nodes)
        Hamiltonian = self.create_Hamiltonian(G)
        return Hamiltonian
