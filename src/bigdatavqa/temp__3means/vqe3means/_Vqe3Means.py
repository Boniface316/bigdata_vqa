import numpy as np
from meansClustering.coresetUtils import Algorithm2, get_bestB
from meansClustering.vqeUtils import (
    approximate_n_trials,
    best_clusters,
    cluster_cost_whole_set,
    data_to_graph,
    random_n_trials,
)
from sklearn.cluster import KMeans


class Vqe_3_Means:
    def __init__(
        self, data, sample_size=10, coreset=None, weights=None, random_sample=None
    ):
        self.data = data
        self.sample_size = sample_size
        self.coreset = coreset
        self.weights = weights
        self.random_sample = random_sample

    def set_coreset_and_weights(self):
        """
        returns the coreset and corresponding weights given the raw data
        """
        B = get_bestB(self.data, num_runs=100, k=3)
        coreset, weights = Algorithm2(self.data, 3, B, self.sample_size)
        self.coreset = np.array(coreset)
        self.weights = np.array(weights)

    def set_graph_representation(self, weights=False):
        """
        sets the networkx graph representation of the provided
        data subset - coreset or random sample.
        """
        if weights:
            self.coreset_graph = data_to_graph(self.coreset, weights=self.weights)
            return

        self.random_sample_graph = data_to_graph(self.random_sample)

    def set_best_cluster_centres_by_brute_force(self, weights=False):
        """
        finds the optimal cluster centres on the coreset by brute force.
        takes the graph representation of the coreset, raw data and coreset
        weights as params
        """
        if weights:
            self.best_coreset_centres = best_clusters(
                self.coreset_graph, self.data, weights=self.weights
            )
            return

        self.best_random_sample_centres = best_clusters(
            self.random_sample_graph, self.data
        )

    def get_3_means_cost(self, centres):
        """
        evaluates the 3-means cost function using the provided cluster centres.
        the data parameter is the dataset to evaluate the cost function on.
        """
        return cluster_cost_whole_set(self.data, centres)

    def get_vqe_bound(self, weights=False):
        """
        computes the bound on the whole dataset
        """
        if weights:
            self.set_best_cluster_centres_by_brute_force(weights=True)
            return self.get_3_means_cost(self.best_coreset_centres)

        self.set_best_cluster_centres_by_brute_force()
        return self.get_3_means_cost(self.best_random_sample_centres)

    def get_vqe_simulation_results(self, depth, num_runs, weights=False):
        if weights:
            return approximate_n_trials(
                self.coreset_graph,
                self.data,
                self.coreset,
                self.weights,
                depth,
                num_runs,
            )

        return random_n_trials(
            self.random_sample_graph, self.data, self.random_sample, num_runs, depth
        )

    def get_classical_cost(data, num_cluster_centres=3):
        """
        uses sci-kit learn library to compute the 3 mean cost
        """
        kmeans = KMeans(n_clusters=num_cluster_centres, random_state=0).fit(data)
        return cluster_cost_whole_set(data, kmeans.cluster_centers_)

    def fit_coreset(self, circuit_depths, num_runs=5, coreset=None, weights=None):
        """
        Approximates the best cluster centres on the coreset.
        The return type is a dictionary with the simulation results
        """
        if (self.coreset == None) and (self.weights == None):
            if (coreset == None) and (weights == None):
                self.set_coreset_and_weights()
            else:
                self.coreset = coreset
                self.weights = weights

        # Generate graph, then find best clusters and compute bound
        self.set_graph_representation(weights=True)

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
            print(
                f"VQE simulation cost for circuit depth {depth}: {simulation_costs[i]}"
            )

        return {
            "coreset": self.coreset,
            "weights": self.weights,
            "circuit depths": circuit_depths,
            "vqe costs": simulation_costs,
            "cluster centres": simulation_centres,
        }
