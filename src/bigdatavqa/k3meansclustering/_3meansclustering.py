from math import pi

import cudaq
import numpy as np
from loguru import logger
from sklearn.cluster import KMeans
from cudaq import spin


from ..coreset import coreset_to_graph, Coreset, normalize_np
from ..optimizer import get_optimizer
from ..vqe_utils import kernel_two_local


def get_coreset_vec_and_weights(
    raw_data,
    number_of_qubits,
    number_of_coresets_to_evaluate,
    number_of_centroids_evaluation,
):
    coreset = Coreset()

    return coreset.get_best_coresets(
        data_vectors=raw_data,
        number_of_runs=number_of_centroids_evaluation,
        coreset_numbers=number_of_qubits,
        size_vec_list=number_of_coresets_to_evaluate,
        use_kmeans_cost=False,
    )


def get_3means_cluster_centers_and_cost(
    coreset_vectors,
    coreset_weights,
    circuit_depth,
    raw_data,
    num_runs=5,
    best_cost=-np.inf,
    best_centers=None,
    normalize=True,
    centralize=True,
):
    coreset_vectors, coreset_weights = get_coreset_vec_and_weights()

    if normalize:
        coreset_vectors, coreset_weights = normalize_np(
            coreset_vectors, centralize=centralize
        ), normalize_np(coreset_weights, centralize=centralize)

    # TODO: coreset_to_graph - use two qubits to represent a node

    coreset_graph, _ = coreset_to_graph(coreset_vectors, coreset_weights)

    for i in range(num_runs):
        cluster_centers = get_3means_clusters_centers(
            coreset_graph, coreset_weights, circuit_depth
        )
        cost_for_clusters = get_3means_cost(raw_data, cluster_centers)
        best_cost, best_centers = get_best_cost_and_centers(
            cost_for_clusters, cluster_centers, best_cost, best_centers
        )

    return best_centers, best_cost


def get_3means_clusters_centers(
    coreset_graph, coreset_vectors, coreset_weights, circuit_depth
):
    # prviusly approx_clusters
    partition = get_approximate_partition(
        coreset_graph, circuit_depth
    )  # [[0], [4, 8], [2, 6]]

    cluster_size = len(coreset_vectors[0])

    clusters_centers = np.array([np.zeros(cluster_size)] * 3)

    # Compute the sum of weights divided by 3

    W = np.sum(coreset_weights) / 3
    # Compute cluster centres
    for i in range(len(partition)):
        for vertex in partition[i]:
            # TODO: clarify why we divide by 2
            weight = (
                coreset_weights[int(vertex) / 2] * coreset_graph.nodes[vertex]["weight"]
            )
            clusters_centers[i] += weight * (1 / W)

    # clusters_centers
    #     array([[ 5.49645877,  1.69655719],
    #    [ 3.5570222 , -0.17831894],
    #    [ 5.7178464 , -0.32430215]])

    return clusters_centers


def get_3means_cost(raw_data, cluster_centers):
    # previusly cluster_cost_whole_set
    center1, center2, center3 = cluster_centers
    cost = 0
    for index, row_data in raw_data.iterrows():
        dist = []
        dist.append(np.linalg.norm(row_data - center1) ** 2)
        dist.append(np.linalg.norm(row_data - center2) ** 2)
        dist.append(np.linalg.norm(row_data - center3) ** 2)
        cost += min(dist)

    return cost


def get_best_cost_and_centers(
    cost_for_clusters, cluster_centers, best_cost, best_centers
):
    if cost_for_clusters < best_cost:
        best_cost = cost_for_clusters
        best_centres = cluster_centers

    return (best_cost, best_centres)


def get_approximate_partition(coreset_graph, circuit_depth):
    """
    Finds approximate partition of the data
    """
    # Simulate VQE to find the aprroximate state
    state = approx_optimal_state(coreset_graph, circuit_depth)
    # Initialise the sets
    s1, s2, s3, sets = [], [], [], []
    # Create list of vertices of G
    vertices = list(coreset_graph.nodes)
    # vertices = [0,2,4,6,8,10]
    # Split bitstring into pairs representing vertices
    pairs = [
        state[i : i + 2] for i in range(0, len(state), 2)
    ]  # pairs - ['00', '11', '10', '01', '10']
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

    sets.append(s1)
    sets.append(s2)
    sets.append(s3)

    return sets  # [[0], [4, 8], [2, 6]]


def approx_optimal_state(
    coreset_graph, circuit_depth, max_iterations=100, max_shots=1024
):
    """
    #Optimises the vqe parameters to approximate an optimal state
    Inputs:
    G: type networkx graph - the problem graph
    depth: type int - the circuit depth to be used
    """
    # Each qubit requires two parameters and there are twice as many
    # qubits as nodes in the graph.
    number_of_qubits = 2 * len(list(coreset_graph.nodes))
    # Using the formula: num_params = 2 * num_qubits * (depth + 1)
    # TODO: verify this works with the current setup - maybe take it from divclustering
    number_of_parameters = 2 * number_of_qubits * (circuit_depth + 1)

    # Find optimal parameters
    optimizer, parameter_count = get_optimizer(
        max_iterations, circuit_depth, number_of_qubits
    )

    Hamiltonian = get_3means_Hamiltonian()

    _, optimal_parameters = cudaq.vqe(
        kernel=kernel_two_local(number_of_qubits, circuit_depth),
        spin_operator=Hamiltonian[0],
        optimizer=optimizer,
        parameter_count=parameter_count,
        shots=max_shots,
    )

    counts = cudaq.sample(
        kernel_two_local(number_of_qubits, circuit_depth),
        optimal_parameters,
        shots_count=max_shots,
    )

    # Find the state that was measured most frequently
    # opt_state return from the original code - i.e 0011100110
    return counts.most_probable()


def get_3means_Hamiltonian(G):
    for i, j in G.edges():
        weight = G[i][j]["weight"]
        H += weight * (
            (5 * spin.i(0) * spin.i(1) * spin.i(2) * spin.i(3))
            + spin.z(1)
            + spin.z(3)
            - (spin.z(0) * spin.z(2))
            - (3 * spin.z(1) * spin.z(3))
            - (spin.z(0) * spin.z(1) * spin.z(2))
            - (spin.z(0) * spin.z(2) * spin.z(3))
            - (spin.z(0) * spin.z(1) * spin.z(2) * spin.z(3))
        )

    return -(1 / 8) * H
