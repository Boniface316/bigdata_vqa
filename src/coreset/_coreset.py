from typing import List

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class Coreset:
    # The codes snippets in this class is taken from the link:
    # https://github.com/teaguetomesh/coresets/blob/ae69df4f52d683c54ab229489e5102b09378da86/kMeans/coreset.py
    def get_coresets(
        self,
        data_vectors: np.ndarray,
        number_of_runs: int,
        coreset_numbers: int,
        size_vec_list: int = 100,
    ):

        B = self.get_bestB(
            data_vectors=data_vectors,
            number_of_runs=number_of_runs,
            k=coreset_numbers,
        )
        coreset_vectors, coreset_weights = [None] * size_vec_list, [
            None
        ] * size_vec_list
        for i in range(size_vec_list):
            coreset_vectors[i], coreset_weights[i] = self.BFL16(
                data_vectors, B=B, m=coreset_numbers
            )

        return [coreset_vectors, coreset_weights]

    def get_bestB(self, data_vectors: np.ndarray, number_of_runs: int, k: int):

        bestB, bestB_cost = None, np.inf

        # pick B with least error from num_runs runs
        for _ in range(number_of_runs):
            B = self.Algorithm1(data_vectors, k=k)
            cost = self.get_cost(data_vectors, B)
            if cost < bestB_cost:
                bestB, bestB_cost = B, cost

        return bestB

    def Algorithm1(self, data_vectors: np.ndarray, k: int):
        B = []
        B.append(data_vectors[np.random.choice(len(data_vectors))])

        for _ in range(k - 1):
            p = np.zeros(len(data_vectors))
            for i, x in enumerate(data_vectors):
                p[i] = self.dist_to_B(x, B) ** 2
            p = p / sum(p)
            B.append(data_vectors[np.random.choice(len(data_vectors), p=p)])

        return B

    def get_cost(self, data_vectors, B):

        cost = 0
        for x in data_vectors:
            cost += self.dist_to_B(x, B) ** 2
        return cost

    def dist_to_B(self, x, B, return_closest_index=False):

        min_dist = np.inf
        closest_index = -1
        for i, b in enumerate(B):
            dist = np.linalg.norm(x - b)
            if dist < min_dist:
                min_dist = dist
                closest_index = i
        if return_closest_index:
            return min_dist, closest_index
        return min_dist

    def BFL16(self, P, B, m):

        num_points_in_clusters = {i: 0 for i in range(len(B))}
        sum_distance_to_closest_cluster = 0
        for p in P:
            min_dist, closest_index = self.dist_to_B(p, B, return_closest_index=True)
            num_points_in_clusters[closest_index] += 1
            sum_distance_to_closest_cluster += min_dist**2

        Prob = np.zeros(len(P))
        for i, p in enumerate(P):
            min_dist, closest_index = self.dist_to_B(p, B, return_closest_index=True)
            Prob[i] += min_dist**2 / (2 * sum_distance_to_closest_cluster)
            Prob[i] += 1 / (2 * len(B) * num_points_in_clusters[closest_index])

        assert 0.999 <= sum(Prob) <= 1.001, (
            "sum(Prob) = %s; the algorithm should automatically "
            "normalize Prob by construction" % sum(Prob)
        )
        chosen_indices = np.random.choice(len(P), size=m, p=Prob)
        weights = [1 / (m * Prob[i]) for i in chosen_indices]

        return [P[i] for i in chosen_indices], weights

    def get_best_coresets(self, data_vectors, coreset_vectors, coreset_weights):

        cost_coreset = [
            self.kmeans_cost(
                data_vectors,
                coreset_vectors=coreset_vectors[i],
                sample_weight=coreset_weights[i],
            )
            for i in range(10)
        ]
        best_index = cost_coreset.index(np.min(cost_coreset))
        best_coreset_vectors = coreset_vectors[best_index]
        best_coreset_weights = coreset_weights[best_index]

        return best_coreset_vectors, best_coreset_weights

    def kmeans_cost(self, data_vectors, coreset_vectors, sample_weight=None):

        kmeans = KMeans(n_clusters=2).fit(coreset_vectors, sample_weight=sample_weight)
        return self.get_cost(data_vectors, kmeans.cluster_centers_)


def gen_coreset_graph(
    coreset_vectors: np.ndarray,
    coreset_weights: np.ndarray,
    metric: str = "dot",
):
    """
    Generate a complete weighted graph using the provided set of coreset points

    Parameters
    ----------
    coreset_weights : ndarray
        np.Coreset weights in array format

    coreset_vectors : ndarray
        Data points of the coreset

    metric : str
        Choose the desired metric for computing the edge weights.
        Options include: dot, dist

    Returns
    -------
    coreset : List((weight, vector))
        The set of points used to construct the graph
    G : NetworkX Graph
        A complete weighted graph
    H : List((coef, pauli_string))
        The equivalent Hamiltonian for the generated graph
    weight_matrix : np.array
        Edge weights of the graph in matrix
    weights : np.array
        Edge weights of the graph in an array

    """

    coreset = [(w, v) for w, v in zip(coreset_weights, coreset_vectors)]

    if coreset is None:
        # Generate a graph instance with sample coreset data
        coreset = []
        # generate 3 points around x=-1, y=-1
        for _ in range(3):
            # use a uniformly random weight
            # weight = np.random.uniform(0.1,5.0,1)[0]
            weight = 1
            vector = np.array(
                [
                    np.random.normal(loc=-1, scale=0.5, size=1)[0],
                    np.random.normal(loc=-1, scale=0.5, size=1)[0],
                ]
            )
            new_point = (weight, vector)
            coreset.append(new_point)

        # generate 3 points around x=+1, y=1
        for _ in range(2):
            # use a uniformly random weight
            # weight = np.random.uniform(0.1,5.0,1)[0]
            weight = 1
            vector = np.array(
                [
                    np.random.normal(loc=1, scale=0.5, size=1)[0],
                    np.random.normal(loc=1, scale=0.5, size=1)[0],
                ]
            )
            new_point = (weight, vector)
            coreset.append(new_point)

    # Generate a networkx graph with correct edge weights
    n = len(coreset)
    G = nx.complete_graph(n)
    H = []
    weights = []
    weight_matrix = np.zeros(len(G.nodes) ** 2).reshape(len(G.nodes()), -1)
    for edge in G.edges():
        pauli_str = ["I"] * n
        # coreset points are labelled by their vertex index
        v_i = edge[0]
        v_j = edge[1]
        pauli_str[v_i] = "Z"
        pauli_str[v_j] = "Z"
        w_i = coreset[v_i][0]
        w_j = coreset[v_j][0]
        if metric == "dot":
            mval = np.dot(coreset[v_i][1], coreset[v_j][1])
        elif metric == "dist":
            mval = np.linalg.norm(coreset[v_i][1] - coreset[v_j][1])
        else:
            raise Exception("Unknown metric: {}".format(metric))

        weight_val = w_i * w_j
        weight_matrix[v_i, v_j] = weight_val
        weight_matrix[v_j, v_i] = weight_val
        G[v_i][v_j]["weight"] = w_i * w_j * mval
        weights.append(w_i * w_j * mval)
        H.append((w_i * w_j * mval, pauli_str))

    return coreset, G, H, weight_matrix, weights


def get_cv_cw(cv: np.ndarray, cw: np.ndarray, idx_vals: int, normalize=True):

    """
    Get the coreset vector and weights from index value of the hierarchy

    Args:
        cv: Coreset vectors
        cw: coreset weights
        idx_vals: Index value in the hierarchy
        normalize: normalize the cv and cw or not

    Returns:
        coreset vectors and weights
    """

    cw = cw[idx_vals]
    cv = cv[idx_vals]

    if normalize:
        cv = normalize_np(cv, True)
        cw = normalize_np(cw)

    return cw, cv


def normalize_np(cv: np.ndarray, centralize=False):

    """
        Normalize and centralize the data

    Args:
        cv: coreset vectors

    Returns:
        normnalized coreset vector
    """

    cv_pd = pd.DataFrame(cv)

    if centralize:
        cv_pd = cv_pd - cv_pd.mean()

    for column in cv_pd.columns:
        cv_pd[column] = cv_pd[column] / cv_pd[column].abs().max()

    cv_norm = cv_pd.to_numpy()

    return cv_norm
