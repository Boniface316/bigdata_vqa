import datetime

import numpy as np
import pandas as pd
import pennylane as qml
import pytz
from divisiveclustering.bruteforceutils import brute_force_cost_2
from divisiveclustering.coresetsUtils import coreset_to_graph
from divisiveclustering.datautils import DataUtils
from divisiveclustering.helpers import add_children_to_hc
from divisiveclustering.quantumutils import (
    create_Hamiltonian_for_K2,
    find_complementary_bitstring,
    get_Hamil_variables,
    get_other_solutions,
    get_probs,
    probs_to_table,
    vqe_ansatz,
)

# ct stores current time
st = datetime.datetime.now()
est_tz = pytz.timezone("US/Eastern")
coreset_numbers = 5
qubits = coreset_numbers
centers = 4
depth = 1
shots = 50
max_iterations = 100
step_size = 0.01
i = 0
single_clusters = 0
threshold = 0.01
file_index = 0

data_utils = DataUtils()

coreset_vectors, coreset_weights, data_vec = data_utils.get_files(
    coreset_numbers, centers
)

coreset_points, G, H, weight_matrix, weights = coreset_to_graph(
    coreset_vectors, coreset_weights, metric="dot"
)

df = pd.DataFrame(coreset_vectors, columns=list("XY"))

df["Name"] = [chr(i + 65) for i in df.index]

index_vals = [i for i in range(len(coreset_weights))]

str_prob_dict = {}


while single_clusters < len(index_vals):
    print(i)
    if i > 0:
        hc = data_utils.load_object("VQE", coreset_numbers, centers, depth, "hc")
    else:
        hc = []
        hc.append(index_vals)

    if len(hc[i]) == 1:
        single_clusters += 1
        i += 1
    else:
        print(datetime.datetime.now())
        index_vals_temp = hc[i]
        new_df = df.iloc[index_vals_temp]
        new_df = new_df.drop(columns=["Name"])
        G, weights, qubits = get_Hamil_variables(
            coreset_vectors, coreset_weights, index_vals_temp, new_df
        )

        print("qubits: " + str(qubits))

        dev = qml.device("qulacs.simulator", wires=qubits, shots=shots)
        cost_H = create_Hamiltonian_for_K2(G, weights, nodes=qubits, add_identity=False)
        params = np.random.random(qubits * 4 * depth)

        opt = qml.GradientDescentOptimizer(step_size)
        cost_fn = qml.ExpvalCost(vqe_ansatz, cost_H, dev)
        nt = datetime.datetime.now()

        if qubits > 6:
            max_iterations = 500
        else:
            max_iterations = 100 * qubits

        for n in range(max_iterations):
            params = opt.step(cost_fn, params, depth=depth, dev=dev, G=G)
            if n % 100 == 0:
                print(n)
                print(datetime.datetime.now() - nt)
                nt = datetime.datetime.now()
            elif n == max_iterations - 1:
                print("done")

        probs = get_probs(params, depth=depth, dev=dev, qubits=qubits, G=G)

        probs_table, _ = probs_to_table(probs, qubits)

        probs_cost_dict_full = brute_force_cost_2(probs_table.solutions, G)

        bitstring_full = max(probs_cost_dict_full, key=probs_cost_dict_full.get)

        if qubits > 2:
            probs_filtered = probs_table[probs_table.probs >= threshold]

        else:
            probs_filtered = probs_table.copy()

        probs_filtered_cost_dict = brute_force_cost_2(probs_filtered.solutions, G)

        bistring_filtered = max(
            probs_filtered_cost_dict, key=probs_filtered_cost_dict.get
        )

        bistring_full_complementary = find_complementary_bitstring(bitstring_full)

        new_df["clusters"] = [int(i) for i in bistring_filtered]

        if len(bistring_filtered) > 2:
            prob_val_full = float(
                probs_table[probs_table.solutions == bitstring_full].probs
            )
            prob_val = float(
                probs_filtered[probs_filtered.solutions == bistring_filtered].probs
            )
            prob_val_complemantry = float(
                probs_table[probs_table.solutions == bistring_full_complementary].probs
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
            "VQE", coreset_numbers, centers, depth, hc, df, str_prob_dict
        )
        data_utils.save_raw_object(
            "VQE", coreset_numbers, centers, depth, probs_table, file_index
        )
        data_utils.save_raw_object(
            "VQE",
            coreset_numbers,
            centers,
            depth,
            probs_cost_dict_full,
            file_index,
            "_cost_dict",
        )

        data_utils.write_i(i, single_clusters)
        print("file saved")
        print(i)

        i += 1
        file_index += 1
