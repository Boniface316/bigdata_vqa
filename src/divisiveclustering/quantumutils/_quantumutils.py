from typing import Dict

import numpy as np
import pandas as pd
import pennylane as qml

from divisiveclustering.bruteforceutils import brute_force_cost_2
from divisiveclustering.coresetsUtils import gen_coreset_graph, get_cv_cw
from divisiveclustering.datautils import DataUtils
from divisiveclustering.helpers import add_children_to_hc


def QAOA_divisive_clustering(
    df: pd.DataFrame,
    cw: np.ndarray,
    cv: np.ndarray,
    coreset_numbers: int = 12,
    centers: int = 5,
    depth: int = 1,
    shots: int = 1000,
    max_iterations: int = 1000,
    step_size: int = 0.01,
    data_folder: int = None,
):

    """
    Performs the entire divisive clustering using QAOA

    Args:
        df: dataframe of coreset vectors
        cw: coreset weights
        cv: coreset vectors
        coreset_numbers: number of coreset
        centers: number of blobs
        depth: depth of circuit
        shots: number of shots
        max_iterations: maximum iteration for optimization
        step_size: step size for optimization
        data_folder: where to save/load files

    Returns:
        Hierarchial clustering of the data
    """
    data_utils = DataUtils(data_folder)
    hc = []
    index_vals = [i for i in range(len(cw))]
    i = 0
    single_clusters = 0
    file_index = 0

    str_prob_dict = {}

    hc.append(index_vals)

    while single_clusters < len(index_vals):
        if i > 0:
            hc = data_utils.load_object("QAOA", coreset_numbers, centers, depth, "hc")
        else:
            hc = []
            hc.append(index_vals)

        if len(hc[i]) == 1:
            single_clusters += 1
            i += 1
        else:

            index_vals_temp = hc[i]
            new_df = df.iloc[index_vals_temp]
            new_df = new_df.drop(columns=["Name"])

            G, weights, qubits = get_Hamil_variables(cv, cw, index_vals_temp, new_df)

            dev = qml.device("qulacs.simulator", wires=qubits, shots=shots)
            cost_H = create_Hamiltonian_for_K2(
                G, weights, nodes=qubits, add_identity=False
            )
            params = np.random.random(qubits * 4 * depth)

            params = optimize_params_qaoa(
                build_qaoa,
                params=params,
                max_iterations=max_iterations,
                step_size=step_size,
                H=cost_H,
                dev=dev,
                G=G,
                depth=depth,
            )

            probs = get_probs(
                params, depth=depth, dev=dev, qubits=qubits, G=G, ansatz="QAOA"
            )

            probs_table, _ = probs_to_table(probs, qubits)

            probs_cost_dict_full = brute_force_cost_2(probs_table.solutions, G)

            bitstring_full = max(probs_cost_dict_full, key=probs_cost_dict_full.get)

            if qubits > 2:
                probs_filtered = probs_table[probs_table.probs >= 0.01]

            else:
                probs_filtered = probs_table.copy()

            probs_filtered_cost_dict = brute_force_cost_2(probs_filtered.solutions, G)

            bistring_filtered = max(
                probs_filtered_cost_dict, key=probs_filtered_cost_dict.get
            )

            bistring_full_complementary = find_complementary_bitstring(bitstring_full)

            new_df["clusters"] = [int(i) for i in bistring_filtered]

            # new_df['clusters'] = qClusUtils.get_clusters(probs, qubits)

            if len(bistring_filtered) > 2:
                prob_val_full = float(
                    probs_table[probs_table.solutions == bitstring_full].probs
                )
                prob_val = float(
                    probs_filtered[probs_filtered.solutions == bistring_filtered].probs
                )
                prob_val_complemantry = float(
                    probs_table[
                        probs_table.solutions == bistring_full_complementary
                    ].probs
                )
                highest_prob = probs_table.iloc[-1]["probs"]
                highest_prob_bs = probs_table.iloc[-1]["solutions"]
                other_solutions = get_other_solutions(probs_cost_dict_full)
                str_prob_dict.update(
                    {
                        bistring_filtered: [
                            prob_val,
                            {
                                highest_prob_bs: [
                                    highest_prob,
                                    probs_cost_dict_full[highest_prob_bs],
                                ]
                            },
                            {
                                bitstring_full: [
                                    prob_val_full,
                                    probs_cost_dict_full[bitstring_full],
                                ]
                            },
                            {
                                bistring_full_complementary: [
                                    prob_val_complemantry,
                                    probs_cost_dict_full[bistring_full_complementary],
                                ]
                            },
                            other_solutions,
                        ]
                    }
                )

            hc = add_children_to_hc(new_df, hc)

            data_utils.save_object(
                "QAOA", coreset_numbers, centers, depth, hc, df, str_prob_dict
            )
            data_utils.save_raw_object(
                "QAOA", coreset_numbers, centers, depth, probs_table, file_index
            )
            data_utils.save_raw_object(
                "QAOA",
                coreset_numbers,
                centers,
                depth,
                probs_cost_dict_full,
                file_index,
                "_cost_dict",
            )

            data_utils.write_i(i, single_clusters)
            print("file is saved")
            print(i)

            i += 1
            file_index += 1

    return hc


def get_Hamil_variables(
    coreset_vectors: np.ndarray,
    coreset_weights: np.ndarray,
    index_vals_temp: int,
    new_df: pd.DataFrame,
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
    cw, cv = get_cv_cw(coreset_vectors, coreset_weights, index_vals_temp)
    coreset_points, G, H, weight_matrix, weights = gen_coreset_graph(
        cv, cw, metric="dot"
    )
    qubits = len(new_df)

    return G, weights, qubits


def create_Hamiltonian_for_K2(G, weights: np.ndarray, nodes, add_identity=False):
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
    if add_identity:
        H = -1 * sum(weights) * qml.Identity(0)
    else:
        H = 0 * qml.Identity(0)

    for i, j in G.edges():
        weight = G[i][j]["weight"][0]
        H += weight * (qml.PauliZ(i) @ qml.PauliZ(j))

    return qml.Hamiltonian(H.coeffs, H.ops)


def problem_ansatz(gamma: float, G):
    """
    Create problem ansatz for QAOA

    Args:
        gamma:  gamma value for the problem
        G: Graph of the problem
    """
    for edge in G.edges():
        wire1 = edge[0]
        wire2 = edge[1]
        phi = 1 * G[wire1][wire2]["weight"]
        qml.CNOT(wires=[wire1, wire2])
        qml.RZ(2 * phi, wires=wire2)
        qml.CNOT(wires=[wire1, wire2])


def mixer_ansatz(beta: float, dev):
    """
    Mixer hamiltonian of QAOA

    Args:
        beta: beta for QAOA
        dev: device information from pennylane
    """

    wires = dev.wires
    for wire in range(len(wires)):
        qml.RX(2 * beta, wires=wire)


def build_qaoa(params: np.ndarray, wires: list, depth: int, G, dev):
    """
    Build a QAOA circuit

    Args:
        params: circuit parameters
        wires: wires in a circuit
        depth: depth of the circuit
        G: Problem as a graph
        dev: device information from pennylane
    """
    params = params.reshape(2, -1)
    gamma = params[0]
    beta = params[1]

    for wire in range(len(dev.wires)):
        qml.Hadamard(wire)

    for p in range(depth):
        problem_ansatz(gamma[p], G)
        mixer_ansatz(beta[p], dev)


def optimize_params_qaoa(
    circuit, params, max_iterations, step_size, H, dev, G, depth, print_at=10
):
    """_summary_

    Args:
        circuit (_type_): _description_
        params (_type_): _description_
        max_iterations (_type_): _description_
        step_size (_type_): _description_
        H (_type_): _description_
        dev (_type_): _description_
        G (_type_): _description_
        depth (_type_): _description_
        print_at (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    opt = qml.GradientDescentOptimizer(stepsize=step_size)
    cost_fn = qml.ExpvalCost(circuit, H, dev)

    qubits = len(H.wires)

    if qubits > 6:
        max_iterations = 1000
    else:
        max_iterations = 100 * qubits

    for n in range(max_iterations):
        params = opt.step(cost_fn, params, dev=dev, depth=depth, G=G)

    print("done")

    return params


def get_probs(params, depth, dev, qubits, G, ansatz="VQE"):

    """_summary_

    Args:
        params (_type_): _description_
        depth (_type_): _description_
        dev (_type_): _description_
        qubits (_type_): _description_
        G (_type_): _description_
        ansatz (str, optional): _description_. Defaults to "VQE".

    Returns:
        _type_: _description_
    """

    probs = [0] * qubits

    if ansatz == "VQE":

        @qml.qnode(dev)
        def probability_circuit(params, depth, dev):
            vqe_ansatz(params, qubits, depth, G, dev)
            return qml.probs(wires=range(qubits))

        probs = probability_circuit(params, depth=depth, dev=dev)
    elif ansatz == "QAOA":

        @qml.qnode(dev)
        def probability_circuit(params, wires, dev=dev, depth=depth, G=G):
            # qUtils.build_qaoa(params, dev.wires, depth, G, dev)
            build_qaoa(params, dev.wires, depth, G, dev)
            return qml.probs(wires=range(len(dev.wires)))

        probs = probability_circuit(params, qubits, dev=dev, depth=depth, G=G)

    else:
        print("unknown ansatz")

    return probs


def probs_to_table(probs: np.ndarray, qubits: int):
    """

    Convert probability to a table

    Args:
        probs: probability outcome
        qubits: number of qubits

    Returns:
       Table with probabilities
    """

    sol_prob_pd = pd.DataFrame({"solutions": range(2**qubits), "probs": probs})
    solutions = sol_prob_pd["solutions"].values

    solution_binary = []
    for sol in solutions:
        str_val = str(bin(sol))
        str_val = str_val.split("b")[1]
        if len(str_val) < qubits:
            number_of_zeros_to_add = qubits - len(str_val)
            str_val = "0" * number_of_zeros_to_add + str_val
        solution_binary.append(str_val)

    sol_prob_pd["solutions"] = np.array(solution_binary)
    sol_prob_pd = sol_prob_pd.drop([0, len(sol_prob_pd) - 1])
    sol_prob_pd = sol_prob_pd.sort_values("probs")

    return sol_prob_pd, sol_prob_pd.iloc[-1, 0]


def find_complementary_bitstring(solution_bitstring: str):
    """
    Find the complementary bitstring of the solution

    Args:
        solution_bitstring: solution bitstring from the circuit

    Returns:
        bitstring that complements the solution
    """

    total_val = 2 ** len(solution_bitstring) - 1

    delta_val = total_val - int(solution_bitstring, 2)

    delta_bin = bin(delta_val)

    delta_bin = delta_bin[2:]

    if len(delta_bin) < len(solution_bitstring):
        diff_bin = len(solution_bitstring) - len(delta_bin)
        delta_bin = "0" * diff_bin + delta_bin

    return delta_bin


def get_other_solutions(probs_filtered_dict_full: Dict):

    """
    Find if there are other solutions

    Args:
        probs_filtered_dict_full: dictionary of probability

    Returns:
        list of alternate solutions
    """

    cost_pd = pd.DataFrame(probs_filtered_dict_full.items(), columns=["string", "cost"])

    cost_pd = cost_pd.sort_values(["cost"])

    max_cost_val = cost_pd.iloc[-1].cost

    other_solutions = cost_pd[cost_pd.cost == max_cost_val[0]].string

    return list(other_solutions)


def vqe_ansatz(params: np.ndarray, wires, depth: int, G, dev):
    """
    Create VQE ansatz for the problem

    Args:
        params: circuit parameters
        wires: wires of the circuit
        depth: circuit depth
        G: Problem as a graph
        dev: device information from pennylane
    """

    wires = len(dev.wires)

    params_p = params.reshape(depth, 4, -1)

    for p in range(depth):
        theta_vals = params_p[p]

        for wire in range(wires):

            qml.RY(theta_vals[0][wire], wires=wire)
            qml.RZ(theta_vals[1][wire], wires=wire)

            if wire > 0:
                qml.CNOT(wires=[wire - 1, wire])

        for wire in range(wires):
            qml.RY(theta_vals[2][wire], wires=wire)
            qml.RZ(theta_vals[3][wire], wires=wire)
