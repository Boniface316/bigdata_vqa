from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Sampling(ABC):
    def __init__(self, method_name) -> None:
        self.sampling_method_name = method_name

    @abstractmethod
    def sample(
        coreset_size: int, raw_data: np.ndarray, *args, **kwargs
    ) -> List[np.ndarray]:
        pass


class D2_sampling(Sampling):
    def __init__(self) -> None:
        super().__init__("D2 sampling")

    def sample(
        coreset_size, raw_data, distance_to_centroids, *args, **kwargs
    ) -> List[np.ndarray]:
        """
        Selects the centroids from the data points using the D2 sampling algorithm.

        Returns:
            List[np.ndarray]: The selected centroids as a list.
        """

        centroids = []
        data_vectors = raw_data

        centroids.append(data_vectors[np.random.choice(len(data_vectors))])

        for _ in range(coreset_size - 1):
            p = np.zeros(len(data_vectors))
            for i, x in enumerate(data_vectors):
                p[i] = distance_to_centroids(x, centroids)[0] ** 2
            p = p / sum(p)
            centroids.append(data_vectors[np.random.choice(len(data_vectors), p=p)])

        return centroids
