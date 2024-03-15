from typing import List, Tuple, Union

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
        coreset_size: int,
        number_of_coresets_to_evaluate: int = 100,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generates coreset vectors and weights using the BFL16 algorithm.

        Args:
            data_vectors (np.ndarray): The input data vectors.
            number_of_runs (int): The number of runs for the D2 samplings to create centroids.
            coreset_size (int): The number of coreset vectors to generate.
            number_of_coresets_to_evaluate (int, optional): The size of the vector list. Defaults to 100.

        Returns:
            Union[List[np.ndarray], List[np.ndarray]]: A list containing the coreset vectors and coreset weights.
        """

        centroids = self.get_best_centroids(
            data_vectors=data_vectors,
            number_of_runs=number_of_runs,
            coreset_size=coreset_size,
        )
        coreset_vectors, coreset_weights = [None] * number_of_coresets_to_evaluate, [
            None
        ] * number_of_coresets_to_evaluate
        for i in range(number_of_coresets_to_evaluate):
            coreset_vectors[i], coreset_weights[i] = self.BFL16(
                data_vectors=data_vectors,
                centroids=centroids,
                coreset_size=coreset_size,
            )

        return [coreset_vectors, coreset_weights]

    def get_best_centroids(
        self, data_vectors: np.ndarray, number_of_runs: int, coreset_size: int
    ) -> List[np.ndarray]:
        """
        Returns the best kmeans++ centroids matrix based on the given data vectors, number of runs, and coreset_size value.

        Parameters:
        data_vectors (np.ndarray): The input data vectors.
        number_of_runs (int): The number of runs to perform.
        k (int): Coreset size.

        Returns:
        np.ndarray: The best kmeans++ centroids matrix.
        """
        best_centroid_coordinates, best_centroid_cost = None, np.inf

        # pick B with least error from num_runs runs
        for _ in range(number_of_runs):
            centroids = self.D2_sampling(data_vectors, coreset_size=coreset_size)
            cost = self.get_cost(data_vectors, centroids)
            if cost < best_centroid_cost:
                best_centroid_coordinates, best_centroid_cost = centroids, cost

        return best_centroid_coordinates

    def D2_sampling(self, data_vectors: np.ndarray, coreset_size: int):
        B = []
        # check if data_vectors is a dataframe. If so, convert it to numpy array
        if isinstance(data_vectors, pd.DataFrame):
            data_vectors = data_vectors.to_numpy()
        B.append(data_vectors[np.random.choice(len(data_vectors))])

        for _ in range(coreset_size - 1):
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

    def BFL16(self, data_vectors, centroids, coreset_size):
        num_points_in_clusters = {i: 0 for i in range(len(centroids))}
        sum_distance_to_closest_cluster = 0
        for p in data_vectors:
            min_dist, closest_index = self.dist_to_B(
                p, centroids, return_closest_index=True
            )
            num_points_in_clusters[closest_index] += 1
            sum_distance_to_closest_cluster += min_dist**2

        Prob = np.zeros(len(data_vectors))
        for i, p in enumerate(data_vectors):
            min_dist, closest_index = self.dist_to_B(
                p, centroids, return_closest_index=True
            )
            Prob[i] += min_dist**2 / (2 * sum_distance_to_closest_cluster)
            Prob[i] += 1 / (2 * len(centroids) * num_points_in_clusters[closest_index])

        assert 0.999 <= sum(Prob) <= 1.001, (
            "sum(Prob) = %s; the algorithm should automatically "
            "normalize Prob by construction" % sum(Prob)
        )
        chosen_indices = np.random.choice(len(data_vectors), size=coreset_size, p=Prob)
        weights = [1 / (coreset_size * Prob[i]) for i in chosen_indices]

        return [data_vectors[i] for i in chosen_indices], weights

    def get_best_coresets(
        self,
        data_vectors,
        number_of_runs,
        coreset_size,
        number_of_coresets_to_evaluate=10,
        use_kmeans_cost=True,
    ):
        if use_kmeans_cost:
            coreset_vectors, coreset_weights = self.get_coresets(
                data_vectors,
                number_of_runs,
                coreset_size,
                number_of_coresets_to_evaluate,
            )

            coreset_vectors, coreset_weights = self.best_coreset_using_kmeans_cost(
                data_vectors, coreset_vectors, coreset_weights
            )
        else:
            B = self.get_best_centroids(
                data_vectors=data_vectors,
                number_of_runs=number_of_runs,
                coreset_size=coreset_size,
            )

            coreset_vectors, coreset_weights = self.Algorithm2(
                data_vectors, B, coreset_size
            )

        return np.array(coreset_vectors), np.array(coreset_weights)

    def kmeans_cost(self, data_vectors, coreset_vectors, sample_weight=None):
        kmeans = KMeans(n_clusters=2).fit(coreset_vectors, sample_weight=sample_weight)
        return self.get_cost(data_vectors, kmeans.cluster_centers_)

    def best_coreset_using_kmeans_cost(
        self, data_vectors, coreset_vectors, coreset_weights
    ):
        cost_coreset = [
            self.kmeans_cost(
                data_vectors,
                coreset_vectors=coreset_vectors[i],
                sample_weight=coreset_weights[i],
            )
            for i in range(10)
        ]

        best_index = cost_coreset.index(np.min(cost_coreset))
        return (coreset_vectors[best_index], coreset_weights[best_index])

    def Algorithm2(
        self,
        data_vectors,
        B,
        coreset_size,
        k=3,
    ):
        alpha = 16 * (np.log2(k) + 2)

        B_i_totals = [0] * len(B)
        B_i = [np.empty_like(data_vectors) for _ in range(len(B))]
        for x in data_vectors:
            _, closest_index = self.dist_to_B(x, B, return_closest_index=True)
            B_i[closest_index][B_i_totals[closest_index]] = x
            B_i_totals[closest_index] += 1

        c_phi = sum([self.dist_to_B(x, B) ** 2 for x in data_vectors]) / len(
            data_vectors
        )

        p = np.zeros(len(data_vectors))

        sum_dist = {i: 0 for i in range(len(B))}
        for i, x in enumerate(data_vectors):
            dist, closest_index = self.dist_to_B(x, B, return_closest_index=True)
            sum_dist[closest_index] += dist**2

        for i, x in enumerate(data_vectors):
            p[i] = 2 * alpha * self.dist_to_B(x, B) ** 2 / c_phi

            _, closest_index = self.dist_to_B(x, B, return_closest_index=True)
            p[i] += (
                4
                * alpha
                * sum_dist[closest_index]
                / (B_i_totals[closest_index] * c_phi)
            )

            p[i] += 4 * len(data_vectors) / B_i_totals[closest_index]
        p = p / sum(p)

        chosen_indices = np.random.choice(len(data_vectors), size=coreset_size, p=p)
        weights = [1 / (coreset_size * p[i]) for i in chosen_indices]

        return [data_vectors[i] for i in chosen_indices], weights


def coreset_to_graph(
    coreset_vectors: np.ndarray,
    coreset_weights: np.ndarray,
    metric: str = "dot",
    number_of_qubits_representing_data: int = 1,
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
        coreset = generate_graph_instance(coreset)

    # Generate a networkx graph with correct edge weights
    vertices = len(coreset)
    vertex_labels = [
        number_of_qubits_representing_data * int(i) for i in range(vertices)
    ]
    G = nx.Graph()
    G.add_nodes_from(vertex_labels)
    edges = [
        (number_of_qubits_representing_data * i, number_of_qubits_representing_data * j)
        for i in range(vertices)
        for j in range(i + 1, vertices)
    ]

    G.add_edges_from(edges)

    weights = []

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
        weights.append(w_i * w_j * mval)

    return G, weights


def generate_graph_instance(coreset):
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

    return coreset


def get_iteration_coreset_vectors_and_weights(
    index_values_to_evaluate,
    initial_coreset_vectors,
    initial_coreset_weights,
    normalize=True,
):

    coreset_weights_for_iteration = initial_coreset_weights[index_values_to_evaluate]
    coreset_vectors_for_iteration = initial_coreset_vectors[index_values_to_evaluate]

    if normalize:

        coreset_vectors_for_iteration = normalize_np(
            coreset_vectors_for_iteration, True
        )
        coreset_weights_for_iteration = normalize_np(coreset_weights_for_iteration)

    return coreset_vectors_for_iteration, coreset_weights_for_iteration


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


def get_coreset_vectors_to_evaluate(
    coreset_vector_df, hierarchial_clustering_sequence, index_iteration_counter
):
    index_values_to_evaluate = hierarchial_clustering_sequence[index_iteration_counter]
    coreset_vectors_to_evaluate = coreset_vector_df.iloc[index_values_to_evaluate]
    return (
        coreset_vectors_to_evaluate.drop(columns=["Name"]),
        index_values_to_evaluate,
    )
