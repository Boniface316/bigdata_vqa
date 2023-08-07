import cudaq
import numpy as np
import warnings
import datetime
import sys
import os
import pickle
from typing import Dict, List
from cudaq import spin
import networkx as nx
from typing import Optional
import pandas as pd
from scipy.stats import multivariate_normal

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


def kernel_two_local(number_of_qubits, layer_count) -> cudaq.Kernel:
    """QAOA ansatz for maxcut"""
    kernel, thetas = cudaq.make_kernel(list)
    qreg = kernel.qalloc(number_of_qubits)

    # Loop over the layers
    theta_position = 0
    
    for i in range(layer_count):

        for j in range(number_of_qubits):
            kernel.rz(thetas[theta_position], qreg[j % number_of_qubits])
            kernel.rx(thetas[theta_position + 1], qreg[j % number_of_qubits])
            kernel.cx(qreg[j], qreg[(j + 1) % number_of_qubits])
            kernel.rz(thetas[theta_position + 2], qreg[j % number_of_qubits])
            kernel.rx(thetas[theta_position + 3], qreg[j % number_of_qubits])
            theta_position += 4

    return kernel

def get_Hamil_variables(
    coreset_vectors: np.ndarray,
    coreset_weights: np.ndarray,
    index_vals_temp: Optional[int] = None,
    new_df: Optional[pd.DataFrame] = None,
):
    """
    Generates the variables required for Hamiltonian

    Args:
        coreset_vectors: Coreset vectors
        coreset_weights: Coreset weights
        index_vals_temp: Index in the hierarchy
        new_df: new dataframe create for this problem,

    Returns:
       Graph, weights and qubits
    """
    if new_df is not None and index_vals_temp is not None:
        coreset_weights, coreset_vectors = get_cv_cw(coreset_vectors, coreset_weights, index_vals_temp)
    
    coreset_points, G, H, weight_matrix, weights = gen_coreset_graph(
        coreset_vectors, coreset_weights, metric="dot"
    )
    qubits = len(G.nodes)

    return G, weights, qubits

def create_Hamiltonian_for_K2(G, qubits, weights: np.ndarray = None,add_identity=False):
    """
    Generate Hamiltonian for k=2

    Args:
        G: Problem as a graph
        weights: Edge weights
        nodes: nodes of the graph
        add_identity: Add identiy or not. Defaults to False.

    Returns:
        _type_: _description_
    """
    H = 0

    for i, j in G.edges():
        weight = G[i][j]["weight"]#[0]
        H += weight * (spin.z(i) * spin.z(j))
        
    return H[0]

class DataUtils:
    def __init__(self, data_folder: str = None, random_seed: int = 1000):
        if data_folder is None:
            self.data_folder = os.getcwd()
        else:
            self.data_folder = data_folder

        self.random_seed = random_seed

    def create_dataset(
        self,
        n_samples: int,
        covariance_values: List[float] = [-0.8, -0.8],
        save_file: bool = True,
        n_features: int = 2,
        number_of_samples_from_distribution: int = 500,
        file_name : str = "data.pickle",
        mean_array: np.ndarray = np.array([[0,0], [7,1]])

    ):
        """
        Create a new data set

        Args:
            n_samples (int): Number of data samples
            centers (int): centers of the data set
            n_features (int): number of dimension of the plot
            random_state (int, optional): random state for reproduction. Defaults to 10.
            save_file (bool, optional): save file or not. Defaults to True.

        Returns:
            created data vector
        """

        random_seed=self.random_seed
    
        X = np.zeros((n_samples,n_features))
    
        # Iterating over different covariance values
        for idx, val in enumerate(covariance_values):
            
            covariance_matrix = np.array([[1, val], [val, 1]])
            
             # Generating a Gaussian bivariate distribution
             # with given mean and covariance matrix
            distr = multivariate_normal(cov = covariance_matrix, mean = mean_array[idx], seed = random_seed)
            
            # Generating 500 samples out of the
            # distribution
            data = distr.rvs(size = number_of_samples_from_distribution)
            
            X[number_of_samples_from_distribution*idx:number_of_samples_from_distribution*(idx+1)][:] = data

        if save_file:
            # check if the data folder exists, if not create it
            if not os.path.exists(self.data_folder + "/data/"):
                os.makedirs(self.data_folder + "/data/")
            # save X as a pickle file in the data folder
            with open(f"{self.data_folder}/{file_name}", 'wb') as handle:
                pickle.dump(X, handle)
                print(f"Data saved in {self.data_folder}/{file_name}")
            
        return X





warnings.filterwarnings("ignore")

time_dict = {}
simulator_options = [None, "custatevec", "custatevec_f32", "cuquantum_mgpu"]
#simulator_options = ["cuquantum"]

number_of_qubits = int(sys.argv[1])
layer_count = 1
parameter_count = 4 * layer_count * number_of_qubits
shots = int(sys.argv[2])


data_utils = DataUtils()

raw_data = data_utils.create_dataset(n_samples=1000, save_file=False)

coresets = Coreset()

coreset_vectors, coreset_weights = coresets.get_coresets(
    data_vectors=raw_data,
    number_of_runs=10,
    coreset_numbers=number_of_qubits,
    size_vec_list=10,
)

best_coreset_vectors, best_coreset_weights = coresets.get_best_coresets(
    raw_data, coreset_vectors, coreset_weights
)

normalized_cv = normalize_np(best_coreset_vectors, centralize=True)
normalized_cw = normalize_np(best_coreset_weights, centralize=False)

coreset_points, G, H, weight_matrix, weights = gen_coreset_graph(
    normalized_cv, normalized_cw, metric="dot"
)

H = create_Hamiltonian_for_K2(G, number_of_qubits, weights)

optimizer = cudaq.optimizers.COBYLA()
optimizer.initial_parameters = np.random.uniform(
    -np.pi / 8.0, np.pi / 8.0, parameter_count
)
print(optimizer.initial_parameters)

# breakpoint()
for simulator in simulator_options:
    if simulator is not None:
        cudaq.set_qpu(simulator)
    print(datetime.datetime.now())
    start_time = datetime.datetime.now()
    optimal_expectation, optimal_parameters = cudaq.vqe(
        kernel=kernel_two_local(number_of_qubits, layer_count),
        spin_operator=H,
        optimizer=optimizer,
        parameter_count=parameter_count,
        shots=shots,
    )

    print(f"end time:{datetime.datetime.now()}")
    print(f"total time:{datetime.datetime.now() - start_time}")
    if simulator is None:
        simulator = "cpu"
    time_dict[simulator] = datetime.datetime.now() - start_time


print(time_dict)