import datetime
import pickle

import numpy as np
import pandas as pd
import pennylane as qml
import pytz
from orqviz.geometric import get_random_normal_vector, get_random_orthonormal_vector
from orqviz.pca import get_pca, perform_2D_pca_scan
from orqviz.scans import perform_1D_interpolation, perform_2D_scan
from orqviz.utils import save_viz_object

from divisiveclustering.coresetsUtils import gen_coreset_graph
from divisiveclustering.datautils import DataUtils
from divisiveclustering.quantumutils import (
    create_Hamiltonian_for_K2,
    get_Hamil_variables,
)

st = datetime.datetime.now()
est_tz = pytz.timezone("US/Eastern")
coreset_numbers = 5
qubits = coreset_numbers
centers = 4
depth = 1
shots = 50
max_iterations = 100
step_size = 0.01
trajectories = 4

trajectories_history_list = []
energy_history_list = []
params_list = []
initial_params_list = []

data_utils = DataUtils()


def problem_ansatz(gamma, G):
    for edge in G.edges():
        wire1 = edge[0]
        wire2 = edge[1]
        phi = 1 * G[wire1][wire2]["weight"]
        qml.CNOT(wires=[wire1, wire2])
        qml.RZ(2 * phi, wires=wire2)
        qml.CNOT(wires=[wire1, wire2])


def mixer_ansatz(beta, dev):
    wires = dev.wires
    for wire in range(len(wires)):
        qml.RX(2 * beta, wires=wire)


def build_qaoa(params, **kwargs):
    params = params.reshape(2, -1)
    gamma = params[0]
    beta = params[1]

    for wire in range(len(dev.wires)):
        qml.Hadamard(wire)

    for p in range(depth):
        problem_ansatz(gamma[p], G)
        mixer_ansatz(beta[p], dev)


coreset_vectors, coreset_weights, data_vec = data_utils.get_files(
    coreset_numbers, centers
)
coreset_points, G, H, weight_matrix, weights = gen_coreset_graph(
    coreset_vectors, coreset_weights, metric="dot"
)

df = pd.DataFrame(coreset_vectors, columns=list("XY"))
index_vals = [i for i in range(len(coreset_weights))]

G, weights, qubits = get_Hamil_variables(
    coreset_vectors, coreset_weights, index_vals, df
)
cost_H = create_Hamiltonian_for_K2(G, weights, nodes=qubits, add_identity=False)
dev = qml.device("qulacs.simulator", wires=cost_H.wires)


cost_fn = qml.ExpvalCost(build_qaoa, cost_H, dev)
opt = qml.GradientDescentOptimizer()


dir1 = get_random_normal_vector(2 * depth)
dir2 = get_random_orthonormal_vector(dir1)

nt = datetime.datetime.now()

for trajectory in range(trajectories):
    print("new trajectory")
    print(datetime.datetime.now().astimezone(est_tz))
    params = np.random.random(depth * 2)
    initial_params = params.copy()
    initial_params_list.append(initial_params)

    gd_param_history = [initial_params]
    gd_cost_history = []

    for n in range(max_iterations):
        params, prev_energy = opt.step_and_cost(
            cost_fn, params, dev=dev, depth=depth, G=G
        )
        gd_param_history.append(params)
        gd_cost_history.append(prev_energy)
        # params = opt.step(cost_fn,params, dev = dev, depth = depth, G =G)
        if (n % 100) == 0:
            print(n)
            print(datetime.datetime.now() - nt)
            nt = datetime.datetime.now()
        elif n == max_iterations - 1:
            print("done")

    params_history_np = np.array(gd_param_history)
    trajectories_history_list.append(params_history_np)
    energy_history_list.append(np.array(gd_cost_history))
    params_list.append(params)
    print("-----")


cost_params_dict = {
    "params": params_list,
    "initial_params": initial_params,
    "trajectories": trajectories_history_list,
    "energy": energy_history_list,
}

file_name = data_utils.data_folder + "/data/QAOA/Orqviz_QAOA_cost_params_dict.pickle"
open_file = open(file_name, "wb")
pickle.dump(cost_params_dict, open_file)


#### 1D plot
end_points = (-5, 5)

initial_params = initial_params_list[0]
params = params_list[0]

interpolation_result = perform_1D_interpolation(
    initial_params, params, cost_fn, end_points=end_points
)

save_viz_object(
    interpolation_result,
    data_utils.data_folder + "/data/QAOA/Orqvoz_QAOA_1D_interpolation_result.pickle",
)


#### 2D plot
scan_2D_result = perform_2D_scan(
    params,
    cost_fn,
    direction_x=dir1,
    direction_y=dir2,
    end_points_x=(-np.pi, np.pi),
    end_points_y=(-np.pi, np.pi),
    n_steps_x=50,
)

save_viz_object(
    scan_2D_result,
    data_utils.data_folder + "/data/QAOA/Orqviz_QAOA_scan_2D_result.pickle",
)


trajectories_collection = np.zeros(2).reshape(-1, 2)

for trajectory in trajectories_history_list:
    trajectories_collection = np.append(trajectories_collection, trajectory, axis=0)

trajectories_collection = np.delete(trajectories_collection, 0, axis=0)

pca = get_pca(trajectories_collection)

scan_pca_result = perform_2D_pca_scan(pca, cost_fn, n_steps_x=40)

save_viz_object(
    scan_pca_result, data_utils.data_folder + "/data/QAOA/Orqviz_QAOA_pca_result.pickle"
)
