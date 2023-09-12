import numpy as np
from loguru import logger
from meansClustering.coresetUtils import Algorithm2, get_bestB
from meansClustering.vqeUtils import (
    approximate_n_trials,
    best_clusters,
    cluster_cost_whole_set,
    data_to_graph,
    random_n_trials,
)
from sklearn.cluster import KMeans

from ..coreset import coreset_to_graph


class VQE3Means:
    def set_best_cluster_centres_by_brute_force(self, weights=False):
        """
        finds the optimal cluster centres on the coreset by brute force.
        takes the graph representation of the coreset, raw data and coreset
        weights as params
        """
        pass

    def get_3_means_cost(self, centres):
        """
        evaluates the 3-means cost function using the provided cluster centres.
        the data parameter is the dataset to evaluate the cost function on.
        """
        pass

    def get_vqe_bound(self, weights=False):
        """
        computes the bound on the whole dataset
        """
        pass

    def get_vqe_simulation_results(self, depth, num_runs, weights=False):
        pass

    def get_classical_cost(data, num_cluster_centres=3):
        """
        uses sci-kit learn library to compute the 3 mean cost
        """
        pass


def cluster3means(
    coreset_vectors,
    coreset_weights,
    circuit_depths,
    num_runs=5,
):
    """
    Approximates the best cluster centres on the coreset.
    The return type is a dictionary with the simulation results
    """
    # Generate graph, then find best clusters and compute bound
    coreset_graph, _ = coreset_to_graph(coreset_vectors, coreset_weights)

    print("Computing VQE bound...")
    vqe_bound = self.get_vqe_bound(weights=True)

    print(f"VQE bound: {vqe_bound}")

    total_depths = len(circuit_depths)
    simulation_costs = np.zeros(total_depths)
    simulation_centres = np.zeros((total_depths, 3))

    # VQE simulations for each circuit depth
    for i, depth in circuit_depths:
        print(f"[{i}/{total_depths}] Simulating with circuit depth {depth}")
        (
            simulation_costs[i],
            simulation_centres[i],
        ) = self.get_vqe_simulation_results(depth, num_runs, weights=True)
        print(f"VQE simulation cost for circuit depth {depth}: {simulation_costs[i]}")

    return {
        "coreset": self.coreset,
        "weights": self.weights,
        "circuit depths": circuit_depths,
        "vqe costs": simulation_costs,
        "cluster centres": simulation_centres,
    }


def best_clusters(coreset_graph, coreset_vectors, coreset_weights):
    """
    Computes best clusters from the optimal partition where we have assumed
    equally weighted clusters W1 = W2 = W3 = W/3
    """
    # Brute force search best partition
    partition = best_partition(G)
    # Get length of feature vectors
    cluster_size = len(coreset_vectors[0])
    # Initialise clusters
    c1 = np.zeros(cluster_size)
    c2 = np.zeros(cluster_size)
    c3 = np.zeros(cluster_size)
    clusters = np.array([c1, c2, c3])
    if weights is None:
        for i in range(len(partition)):
            for vertex in partition[i]:
                weight = G.nodes[vertex]["feature_vector"]
                clusters[i] += weight * (1 / len(partition[i]))
    else:
        # Compute the sum of weights divided by 3
        W = np.sum(weights) / 3
        # Compute cluster centres
        for i in range(len(partition)):
            for vertex in partition[i]:
                weight = weights[int(vertex / 2)] * G.nodes[vertex]["feature_vector"]
                clusters[i] += weight * (1 / W)

    return clusters


def best_partition(G):
    """
    Maps the best state into the best partition
    """
    # Initialise the sets
    s1 = []
    s2 = []
    s3 = []
    # Create list of vertices of G
    vertices = list(G.nodes)
    # Brute force the optimal state
    (_, state) = find_optimal_state(G)
    # Split bitstring into pairs representing vertices
    pairs = [state[i : i + 2] for i in range(0, len(state), 2)]
    # Check vertices for which set they correspond to
    for i, vertex in enumerate(pairs):
        if vertex == "00":
            s1.append(vertices[i])
        elif vertex == "10":
            s2.append(vertices[i])
        elif vertex == "01":
            s3.append(vertices[i])
        elif vertex == "11":
            s3.append(vertices[i])

    sets = []
    sets.append(s1)
    sets.append(s2)
    sets.append(s3)

    return sets
