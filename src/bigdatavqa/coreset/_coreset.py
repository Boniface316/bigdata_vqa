import pickle
from typing import List, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans


# TODO: refactor with abstract class
class Coreset(ABC):
    # https://github.com/teaguetomesh/coresets/blob/ae69df4f52d683c54ab229489e5102b09378da86/kMeans/coreset.py
    def __init__(
        self,
        raw_data: np.ndarray,
        number_of_sampling_for_centroids: int,
        coreset_size: int,
        number_of_coresets_to_evaluate: int,
        sampling_method: Callable,
        coresets_method: str,
    ) -> None:
        self._raw_data = raw_data
        self._coreset_size = coreset_size
        self._number_of_coresets_to_evaluate = number_of_coresets_to_evaluate
        self._number_of_sampling_for_centroids = number_of_sampling_for_centroids
        self._sampling_method = sampling_method
        self._coreset_method = coresets_method

    @property
    def raw_data(self) -> np.ndarray:
        return self._raw_data

    @property
    def coreset_size(self) -> int:
        return self._coreset_size

    @property
    def number_of_coresets_to_evaluate(self) -> int:
        return self._number_of_coresets_to_evaluate

    @property
    def number_of_sampling_for_centroids(self) -> int:
        return self._number_of_sampling_for_centroids

    @property
    def coreset_method(self) -> str:
        return self._coreset_method

    @property
    def sampling_method_name(self) -> Callable:
        return self.sampling_method.sampling_method_name

    @raw_data.setter
    def raw_data(self, raw_data: np.ndarray) -> None:
        self._raw_data = raw_data

    @coreset_size.setter
    def coreset_size(self, coreset_size: int) -> None:
        self._coreset_size = coreset_size

    @number_of_coresets_to_evaluate.setter
    def number_of_coresets_to_evaluate(
        self, number_of_coresets_to_evaluate: int
    ) -> None:
        self._number_of_coresets_to_evaluate = number_of_coresets_to_evaluate

    @number_of_sampling_for_centroids.setter
    def number_of_sampling_for_centroids(
        self, number_of_sampling_for_centroids: int
    ) -> None:
        self._number_of_sampling_for_centroids = number_of_sampling_for_centroids

    @coreset_method.setter
    def coreset_method(self, coreset_method: str) -> None:
        self._coreset_method = coreset_method

    def _get_coresets(
        self, centroids: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generates coreset vectors and weights using the BFL2 algorithm.

        Args:
            centroids (List[np.ndarray]): The centroids to use for the coreset generation.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: List of coreset vectors and weights.
        """

        coreset_vectors_list = []
        coreset_weights_list = []
        for i in range(self.number_of_coresets_to_evaluate):
            coreset_vectors, coreset_weights = (
                self._get_coresets_using_selected_approach(centroids=centroids)
            )
            coreset_vectors_list.append(coreset_vectors)
            coreset_weights_list.append(coreset_weights)

        return (coreset_vectors_list, coreset_weights_list)

    @abstractmethod
    def _get_coresets_using_selected_approach(
        self, centroids: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[float]]:
        pass

    def get_best_coresets(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the best coreset vectors and weights for a given data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The coreset vectors and weights.
        """

        centroids = self.get_best_centroids()

        coreset_vectors, coreset_weights = self._get_coresets(centroids)

        coreset_vectors, coreset_weights = self.best_coreset_using_kmeans_cost(
            coreset_vectors, coreset_weights
        )

        self.coreset_vectors = np.array(coreset_vectors)
        self.coreset_weights = np.array(coreset_weights)

        return (self.coreset_vectors, self.coreset_weights)

    def get_best_centroids(self) -> List[np.ndarray]:
        """
        Get the best centroids using the D2 sampling algorithm.

        Returns:
            List[np.ndarray]: The best centroids.

        """

        best_centroid_coordinates, best_centroid_cost = None, np.inf

        for _ in range(self._number_of_sampling_for_centroids):
            centroids = self._sampling_method.sample(
                self.coreset_size, self.raw_data, self.distance_to_centroids
            )
            cost = self.get_cost(centroids)
            if cost < best_centroid_cost:
                best_centroid_coordinates, best_centroid_cost = centroids, cost

        return best_centroid_coordinates

    def get_cost(self, centroids: Union[List[np.ndarray], np.ndarray]) -> float:
        """
        Computes the sum of between each data points and each centroids.

        Args:
            centroids (Union[List[np.ndarray], np.ndarray]): The centroids to evaluate.

        Returns:
            float: The cost of the centroids.

        """

        cost = 0.0
        for x in self.raw_data:
            cost += self.distance_to_centroids(x, centroids)[0] ** 2
        return cost

    def distance_to_centroids(
        self, data_instance: np.ndarray, centroids: Union[List[np.ndarray], np.ndarray]
    ) -> Tuple[float, int]:
        """
        Compute the distance between a data instance and the centroids.

        Args:
            data_instance (np.ndarray): The data instance.
            centroids (Union[List[np.ndarray], np.ndarray]): The centroids as a list or numpy array.

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

        return (minimum_distance, closest_index)

    def best_coreset_using_kmeans_cost(
        self, coreset_vectors: List[np.ndarray], coreset_weights: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the best coreset using kmeans cost.

        Args:
            coreset_vectors (List[np.ndarray]): The coreset vectors.
            coreset_weights (List[np.ndarray]): The coreset weights.

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

    def kmeans_cost(
        self, coreset_vectors: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> float:
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

    @staticmethod
    def coreset_to_graph(
        coreset_vectors: np.ndarray,
        coreset_weights: np.ndarray,
        metric: Optional[str] = "dot",
        number_of_qubits_representing_data: Optional[int] = 1,
    ) -> nx.Graph:
        """
        Convert coreset vectors to a graph.

        Args:
            coreset_vectors (np.ndarray): The coreset vectors.
            coreset_weights (np.ndarray): The coreset weights.
            metric (str, optional): The metric to use. Defaults to "dot".
            number_of_qubits_representing_data (int, optional): The number of qubits representing the data. Defaults to 1.

        Returns:
            nx.Graph: The graph.
        """

        coreset = [(w, v) for w, v in zip(coreset_weights, coreset_vectors)]

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
            v_i = edge[0] // number_of_qubits_representing_data
            v_j = edge[1] // number_of_qubits_representing_data
            w_i = coreset[v_i][0]
            w_j = coreset[v_j][0]
            if metric == "dot":
                mval = np.dot(
                    coreset[v_i][1],
                    coreset[v_j][1],
                )
            elif metric == "dist":
                mval = np.linalg.norm(coreset[v_i][1] - coreset[v_j][1])
            else:
                raise Exception("Unknown metric: {}".format(metric))

            G[edge[0]][edge[1]]["weight"] = w_i * w_j * mval

        return G

    @staticmethod
    def normalize_array(vectors: np.ndarray, centralize: bool = False) -> np.ndarray:
        """
        Normalize and centralize the array

        Args:
            vectors (np.ndarray): The vectors to normalize
            centralize (bool, optional): Centralize the array. Defaults to False.

        Returns:
            np.ndarray: The normalized array
        """

        if centralize:
            vectors = vectors - np.mean(vectors, axis=0)

        max_abs = np.max(np.abs(vectors), axis=0)
        vectors_norm = vectors / max_abs

        return vectors_norm

    @staticmethod
    def create_dataset(
        n_samples: float,
        covariance_values: List[float] = [-0.8, -0.8],
        n_features: Optional[int] = 2,
        number_of_samples_from_distribution: Optional[int] = 500,
        mean_array: Optional[np.ndarray] = np.array([[0, 0], [7, 1]]),
        random_seed: Optional[int] = 10,
    ) -> np.ndarray:
        """
        Create a dataset with the given parameters.

        Args:
            n_samples (float): The number of samples.
            covariance_values (List[float], optional): The covariance values. Defaults to [-0.8, -0.8].
            n_features (int, optional): The number of features. Defaults to 2.
            number_of_samples_from_distribution (int, optional): The number of samples from the distribution. Defaults to 500.
            mean_array (np.ndarray, optional): The mean array. Defaults to np.array([[0, 0], [7, 1]]).
            random_seed (int, optional): The random seed. Defaults to 10.

        Returns:
            np.ndarray: The dataset created
        """

        random_seed = random_seed

        X = np.zeros((n_samples, n_features))

        for idx, val in enumerate(covariance_values):
            covariance_matrix = np.array([[1, val], [val, 1]])

            distr = multivariate_normal(
                cov=covariance_matrix, mean=mean_array[idx], seed=random_seed
            )

            data = distr.rvs(size=number_of_samples_from_distribution)

            X[
                number_of_samples_from_distribution
                * idx : number_of_samples_from_distribution * (idx + 1)
            ][:] = data

        return X

    @staticmethod
    def load_coreset_dataset(file_name: str = "dataset.pickle"):
        """
        Load a dataset

        Args:
            file_name (str, optional): file name of the dataset. Defaults to "dataset.pickle".

        Returns:
            loaded dataset
        """
        with open(file_name, "rb") as handle:
            X = pickle.load(handle)
            print(f"Data loaded from {file_name}")
        return X
