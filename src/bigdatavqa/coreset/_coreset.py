from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class Coreset:
    # The codes snippets in this class is taken from the link:
    # https://github.com/teaguetomesh/coresets/blob/ae69df4f52d683c54ab229489e5102b09378da86/kMeans/coreset.py
    def __init__(
        self,
        raw_data: np.ndarray,
        number_of_sampling_for_centroids: int,
        coreset_size: int,
        number_of_coresets_to_evaluate: int = 10,
        coreset_method: str = "BFL16",
        k_value_for_algorithm_2: int = 2,
    ) -> None:

        self._raw_data = raw_data
        self._coreset_size = coreset_size
        self._number_of_coresets_to_evaluate = number_of_coresets_to_evaluate
        self._number_of_sampling_for_centroids = number_of_sampling_for_centroids
        self._k_value_for_algorithm_2 = k_value_for_algorithm_2

        if coreset_method not in ["BFL16", "Algorithm2"]:
            raise ValueError("Coreset method must be either BFL16 or Algorithm2.")
        else:
            self._coreset_method = coreset_method

    @property
    def raw_data(self):
        return self._raw_data

    @property
    def coreset_size(self):
        return self._coreset_size

    @property
    def number_of_coresets_to_evaluate(self):
        return self._number_of_coresets_to_evaluate

    @property
    def number_of_sampling_for_centroids(self):
        return self._number_of_sampling_for_centroids

    @property
    def coreset_method(self):
        return self._coreset_method

    @property
    def k_value_for_algorithm_2(self):
        return self._k_value_for_algorithm_2

    @raw_data.setter
    def raw_data(self, raw_data):
        if not isinstance(raw_data, np.ndarray):
            raise ValueError("Data must be a numpy array.")
        self._raw_data = raw_data

    @coreset_size.setter
    def coreset_size(self, coreset_size):
        if coreset_size < 1 or not isinstance(coreset_size, int):
            raise ValueError("Coreset size must be greater than 0 or an integer.")
        self._coreset_size = coreset_size

    @number_of_coresets_to_evaluate.setter
    def number_of_coresets_to_evaluate(self, number_of_coresets_to_evaluate):
        if number_of_coresets_to_evaluate < 1 or not isinstance(
            number_of_coresets_to_evaluate, int
        ):
            raise ValueError("Number of coreset to evaluate must be greater than 0.")
        self._number_of_coresets_to_evaluate = number_of_coresets_to_evaluate

    @number_of_sampling_for_centroids.setter
    def number_of_sampling_for_centroids(self, number_of_sampling_for_centroids):
        if number_of_sampling_for_centroids < 1 or not isinstance(
            number_of_sampling_for_centroids, int
        ):
            raise ValueError("Number of sampling for centroids must be greater than 0.")
        self._number_of_sampling_for_centroids = number_of_sampling_for_centroids

    @coreset_method.setter
    def coreset_method(self, coreset_method):
        if coreset_method not in ["BFL16", "Algorithm2"]:
            raise ValueError("Coreset method must be either BFL16 or Algorithm2.")
        self._coreset_method = coreset_method

    @k_value_for_algorithm_2.setter
    def k_value_for_algorithm_2(self, k_value_for_algorithm_2):
        if k_value_for_algorithm_2 < 2 or not isinstance(k_value_for_algorithm_2, int):
            raise ValueError("K value for algorithm 2 must be greater than 1.")
        self._k_value_for_algorithm_2 = k_value_for_algorithm_2

    def get_best_coresets(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the best coreset vectors and weights for a given data

        Returns:
            Tuple[np.ndarray, np.ndarray]: The coreset vectors and weights.
        """

        centroids = self.get_best_centroids()

        if self._coreset_method == "BFL16":
            print("Using BFL16 method to generate coresets")
            coreset_vectors, coreset_weights = self.get_coresets_using_BFL16(centroids)

        elif self._coreset_method == "Algorithm2":
            print("Using Algorithm2 method to generate coresets")
            coreset_vectors, coreset_weights = self.get_coresets_using_Algorithm2(
                centroids
            )
        else:
            raise ValueError("Coreset method must be either BFL16 or Algorithm2.")

        coreset_vectors, coreset_weights = self.best_coreset_using_kmeans_cost(
            coreset_vectors, coreset_weights
        )

        self.coreset_vectors = coreset_vectors
        self.coreset_weights = coreset_weights

        return np.array(coreset_vectors), np.array(coreset_weights)

    def get_coresets_using_BFL16(
        self, centroids
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generates coreset vectors and weights using the BFL16 algorithm.

        Args:
            centroids (List[np.ndarray]): The centroids to use for the coreset generation.

        Returns:
            Union[List[np.ndarray], List[np.ndarray]]: A list containing the coreset vectors and coreset weights.
        """

        coreset_vectors_list = []
        coreset_weights_list = []
        for i in range(self.number_of_coresets_to_evaluate):
            coreset_vectors, coreset_weights = self.BFL16(centroids=centroids)
            coreset_vectors_list.append(coreset_vectors)
            coreset_weights_list.append(coreset_weights)

        return coreset_vectors_list, coreset_weights_list

        # return [coreset_vectors, coreset_weights]

    def get_best_centroids(self) -> List[np.ndarray]:
        """
        Get the best centroids using the D2 sampling algorithm.

        Returns:
            List[np.ndarray]: The best centroids.

        """

        best_centroid_coordinates, best_centroid_cost = None, np.inf

        for _ in range(self.number_of_sampling_for_centroids):
            centroids = self.D2_sampling()
            cost = self.get_cost(centroids)
            if cost < best_centroid_cost:
                best_centroid_coordinates, best_centroid_cost = centroids, cost

        return best_centroid_coordinates

    def D2_sampling(self) -> List[np.ndarray]:
        """
        Selects the centroids from the data points using the D2 sampling algorithm.

        Returns:
            List[np.ndarray]: The selected centroids as a list.
        """
        centroids = []
        data_vectors = self.raw_data

        centroids.append(data_vectors[np.random.choice(len(data_vectors))])

        for _ in range(self.coreset_size - 1):
            p = np.zeros(len(data_vectors))
            for i, x in enumerate(data_vectors):
                p[i] = self.distance_to_centroids(x, centroids)[0] ** 2
            p = p / sum(p)
            centroids.append(data_vectors[np.random.choice(len(data_vectors), p=p)])

        return centroids

    def get_cost(self, centroids) -> float:
        """
        Computes the sum of between each data points and each centroids.

        Args:
            centroids (List): The centroids to evaluate.

        Returns:
            float: The cost of the centroids.

        """
        cost = 0.0
        for x in self.raw_data:
            cost += self.distance_to_centroids(x, centroids)[0] ** 2
        return cost

    def distance_to_centroids(
        self, data_instance: np.ndarray, centroids: List
    ) -> Tuple[float, int]:
        """
        Compute the distance between a data instance and the centroids.

        Args:
            data_instance (np.ndarray): The data instance.
            centroids (np.ndarray): The centroids.

        Returns:
            Tuple[float, int]: The minimum distance and the index of the closest centroid.
        """
        minimum_distance = np.inf
        closest_index = -1
        for i, centroid in enumerate(centroids):
            distance_between_data_instance_and_centroid = np.linalg.norm(
                data_instance - centroid
            )
            if distance_between_data_instance_and_centroid < minimum_distance:
                minimum_distance = distance_between_data_instance_and_centroid
                closest_index = i

        return minimum_distance, closest_index

    def BFL16(self, centroids) -> Tuple[List[np.ndarray], List[float]]:
        """
        Performs Algorithm 2 from https://arxiv.org/pdf/1612.00889.pdf [BFL16]. This will pick the coreset vectors and its corresponding weights.

        Args:
            centroids (List): The centroids to use for the coreset generation.

        Returns:
            Tuple[List, List]: The coreset vectors and coreset weights.
        """
        number_of_data_points_close_to_a_cluster = {i: 0 for i in range(len(centroids))}
        sum_distance_to_closest_cluster = 0.0
        for data_instance in self.raw_data:
            min_dist, closest_index = self.distance_to_centroids(
                data_instance, centroids
            )
            number_of_data_points_close_to_a_cluster[closest_index] += 1
            sum_distance_to_closest_cluster += min_dist**2

        Prob = np.zeros(len(self._raw_data))
        for i, p in enumerate(self._raw_data):
            min_dist, closest_index = self.distance_to_centroids(p, centroids)
            Prob[i] += min_dist**2 / (2 * sum_distance_to_closest_cluster)
            Prob[i] += 1 / (
                2
                * len(centroids)
                * number_of_data_points_close_to_a_cluster[closest_index]
            )

        if not (0.999 <= sum(Prob) <= 1.001):
            raise ValueError(
                "sum(Prob) = %s; the algorithm should automatically "
                "normalize Prob by construction" % sum(Prob)
            )
        chosen_indices = np.random.choice(
            len(self._raw_data), size=self._coreset_size, p=Prob
        )
        weights = [1 / (self._coreset_size * Prob[i]) for i in chosen_indices]

        return [self._raw_data[i] for i in chosen_indices], weights

    def kmeans_cost(self, coreset_vectors, sample_weight=None) -> float:
        """
        Compute the cost of coreset vectors using kmeans clustering.

        Args:
            coreset_vectors (np.ndarray): The coreset vectors.
            sample_weight (np.ndarray): The sample weights.

        Returns:
            float: The cost of the kmeans clustering.

        """
        kmeans = KMeans(n_clusters=2).fit(coreset_vectors, sample_weight=sample_weight)
        return self.get_cost(kmeans.cluster_centers_)

    def best_coreset_using_kmeans_cost(
        self, coreset_vectors, coreset_weights
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the best coreset using kmeans cost.

        Args:
            coreset_vectors (List): The coreset vectors.
            coreset_weights (List): The coreset weights.

        Returns:
            Tuple: The best coreset vectors and coreset weights.

        """
        cost_coreset = [
            self.kmeans_cost(
                coreset_vectors=coreset_vectors[i],
                sample_weight=coreset_weights[i],
            )
            for i in range(self._number_of_coresets_to_evaluate)
        ]

        best_index = cost_coreset.index(np.min(cost_coreset))
        return (coreset_vectors[best_index], coreset_weights[best_index])

    def get_coresets_using_Algorithm2(
        self, centroids
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Generates coreset vectors and weights using Algorithm 2.

        Args:
            centroids (List): The centroids to use for the coreset generation.

        Returns:
            Tuple[List, List]: The coreset vectors and coreset weights.
        """
        coreset_vectors_list = []
        coreset_weights_list = []
        for i in range(self.number_of_coresets_to_evaluate):
            coreset_vectors, coreset_weights = self.Algorithm2(centroids=centroids)
            coreset_vectors_list.append(coreset_vectors)
            coreset_weights_list.append(coreset_weights)

        return coreset_vectors_list, coreset_weights_list

    def Algorithm2(
        self,
        centroids: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Performs Algorithm 2 from  https://arxiv.org/pdf/1703.06476.pdf.

        Args:
            centroids (List): The centroids to use for the coreset generation.

        Returns:
            Tuple[List, List]: The coreset vectors and coreset weights.
        """
        alpha = 16 * (np.log2(self._k_value_for_algorithm_2) + 2)

        B_i_totals = [0] * len(centroids)
        B_i = [np.empty_like(self._raw_data) for _ in range(len(centroids))]
        for data_instance in self._raw_data:
            _, closest_index = self.distance_to_centroids(data_instance, centroids)
            B_i[closest_index][B_i_totals[closest_index]] = data_instance
            B_i_totals[closest_index] += 1

        c_phi = sum(
            [
                self.distance_to_centroids(data_instance, centroids)[0] ** 2
                for data_instance in self._raw_data
            ]
        ) / len(self._raw_data)

        p = np.zeros(len(self._raw_data))

        sum_dist = {i: 0.0 for i in range(len(centroids))}
        for i, data_instance in enumerate(self._raw_data):
            dist, closest_index = self.distance_to_centroids(data_instance, centroids)
            sum_dist[closest_index] += dist**2

        for i, data_instance in enumerate(self._raw_data):
            p[i] = (
                2
                * alpha
                * self.distance_to_centroids(data_instance, centroids)[0] ** 2
                / c_phi
            )

            closest_index = self.distance_to_centroids(data_instance, centroids)[1]
            p[i] += (
                4
                * alpha
                * sum_dist[closest_index]
                / (B_i_totals[closest_index] * c_phi)
            )

            p[i] += 4 * len(self._raw_data) / B_i_totals[closest_index]
        p = p / sum(p)

        chosen_indices = np.random.choice(
            len(self._raw_data), size=self._coreset_size, p=p
        )
        weights = [1 / (self._coreset_size * p[i]) for i in chosen_indices]

        return [self._raw_data[i] for i in chosen_indices], weights

    @staticmethod
    def coreset_to_graph(
        coreset_vectors: np.ndarray,
        coreset_weights: np.ndarray,
        metric: Optional[str] = "dot",
        number_of_qubits_representing_data: Optional[int] = 1,
    ) -> nx.Graph:
        """
        Generate a complete weighted graph using the provided set of coreset points

        Args:
            metric(str): Choose the desired metric for computing the edge weights.
                         Options include: dot, dist

            number_of_qubits_representing_data (int) : number of qubits representing a data point

        Returns:
            G (nx.Graph) : A complete weighted graph
        """

        coreset = [(w, v) for w, v in zip(coreset_weights, coreset_vectors)]

        # Generate a networkx graph with correct edge weights
        vertices = len(coreset)
        vertex_labels = [
            number_of_qubits_representing_data * int(i) for i in range(vertices)
        ]
        G = nx.Graph()
        G.add_nodes_from(vertex_labels)
        edges = [
            (
                number_of_qubits_representing_data * i,
                number_of_qubits_representing_data * j,
            )
            for i in range(vertices)
            for j in range(i + 1, vertices)
        ]

        G.add_edges_from(edges)

        for edge in G.edges():
            v_i = edge[0]
            v_j = edge[1]
            w_i = coreset[v_i // number_of_qubits_representing_data][0]
            w_j = coreset[v_j // number_of_qubits_representing_data][0]
            if metric == "dot":
                mval = np.dot(
                    coreset[v_i // number_of_qubits_representing_data][1],
                    coreset[v_j // number_of_qubits_representing_data][1],
                )
            elif metric == "dist":
                mval = np.linalg.norm(coreset[v_i][1] - coreset[v_j][1])
            else:
                raise Exception("Unknown metric: {}".format(metric))

            G[v_i][v_j]["weight"] = w_i * w_j * mval

        return G

    @staticmethod
    def normalize_array(vectors: np.ndarray, centralize=False):
        """
        Normalize and centralize the data

        Args:
            vectors: coreset vectors

        Returns:
            normalized coreset vector
        """

        if centralize:
            vectors = vectors - np.mean(vectors, axis=0)

        max_abs = np.max(np.abs(vectors), axis=0)
        vectors_norm = vectors / max_abs

        return vectors_norm
