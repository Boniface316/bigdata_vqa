from typing import List

import networkx as nx
import numpy as np
import pandas as pd


def get_distance_between_two_vectors(
    vector1: np.ndarray, vector2: np.ndarray, weight: np.ndarray
) -> float:
    return weight * np.linalg.norm(vector1 - vector2)


def get_k_means_accumulative_cost(k, clusters, data, data_weights=None):
    accumulativeCost = 0
    currentCosts = np.repeat(0, k)
    data_weights = np.repeat(1, len(data)) if data_weights is None else data_weights
    for vector in data:
        currentCosts = list(
            map(
                get_distance_between_two_vectors,
                clusters,
                np.repeat(vector, k, axis=0),
                data_weights,
            )
        )
        accumulativeCost = accumulativeCost + min(currentCosts)

    return accumulativeCost
