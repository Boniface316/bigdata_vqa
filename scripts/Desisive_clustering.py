import argparse
import warnings

import cudaq
import pandas as pd
from bigdatavqa.coreset import Coreset
from bigdatavqa.divisiveclustering import (
    DivisiveClusteringKMeans,
    DivisiveClusteringMaxCut,
    DivisiveClusteringRandom,
    DivisiveClusteringVQA,
)
from bigdatavqa.optimizer import get_optimizer_for_QAOA, get_optimizer_for_VQE
from bigdatavqa.vqe_utils import get_K2_Hamiltonian, get_QAOA_circuit, get_VQE_circuit

warnings.filterwarnings("ignore")

argparser = argparse.ArgumentParser()

argparser.add_argument(
    "-c",
    "--clustering_methods",
    type=str,
    nargs="+",
    help="Clustering methods to evaluate.",
    default=["QAOA"],
)

argparser.add_argument(
    "-t",
    "--target",
    type=str,
    default="qpp-cpu",
    choices=["qpp-cpu", "nvidia", "nvidia-mgpu"],
    help="Quantum simulator backend. Default is qpp-cpu. See https://nvidia.github.io/cuda-quantum/0.6.0/using/simulators.html for more options.",
)

argparser.add_argument(
    "-d",
    "--depth",
    type=int,
    default=1,
    help="Depth of the QAOA circuit. Default is 1.",
)
argparser.add_argument(
    "-i",
    "--max_iterations",
    type=int,
    default=75,
    help="Max iterations for the optimizer.",
)
argparser.add_argument(
    "-s", "--max_shots", type=int, default=100000, help="Max shots for the simulation."
)
argparser.add_argument("-m", "--M", type=int, default=10, help="Size of the coreset.")

argparser.add_argument(
    "-n", "--N", type=int, default=10000, help="Number of rows for the raw dataset"
)

argparser.add_argument(
    "--number_of_sampling_for_centroids",
    type=int,
    default=10,
    help="Number of sampling for the centroids.",
)

argparser.add_argument(
    "--number_of_coresets_to_evaluate",
    type=int,
    default=10,
    help="Number of coresets to evaluate.",
)

argparser.add_argument(
    "--coreset_method",
    type=str,
    default="BFL2",
    choices=["BFL2", "BLK2"],
    help="Method for the coreset creation.",
)

argparser.add_argument(
    "--normalize_vectors",
    type=bool,
    default=True,
    help="Normalize the vectors.",
)

argparser.add_argument(
    "--max_cutoff_threshold",
    type=float,
    default=0.5,
    help="Max cutoff threshold for the divisive clustering.",
)

argparser.add_argument(
    "--coreset_to_graph_metric",
    type=str,
    default="dist",
    choices=["dist", "dot"],
    help="Metric to calculate the weight of the graph.",
)

argparser.add_argument(
    "--divisive_method",
    nargs="+",
    type=int,
    help="Divisive clustering methods.",
)

args = argparser.parse_args()

number_of_rows = args.N
target = args.target
coreset_size = args.M
circuit_depth = args.depth
max_iterations = args.max_iterations
max_shots = args.max_shots
max_cutoff_threshold = args.max_cutoff_threshold
coreset_to_graph_metric = args.coreset_to_graph_metric
methods = args.clustering_methods


number_of_sampling_for_centroids = args.number_of_sampling_for_centroids
number_of_coresets_to_evaluate = args.number_of_coresets_to_evaluate
coreset_method = args.coreset_method
normalize_vectors = args.normalize_vectors


def create_coreset_df(
    number_of_rows,
    coreset_size,
    coreset_method,
    number_of_coresets_to_evaluate,
    number_of_sampling_for_centroids,
):
    raw_data = Coreset.create_dataset(number_of_rows)
    coreset = Coreset(
        raw_data=raw_data,
        number_of_sampling_for_centroids=number_of_sampling_for_centroids,
        coreset_size=coreset_size,
        number_of_coresets_to_evaluate=number_of_coresets_to_evaluate,
        coreset_method=coreset_method,
    )

    coreset_vectors, coreset_weights = coreset.get_best_coresets()

    coreset_df = pd.DataFrame(
        {
            "X": coreset_vectors[:, 0],
            "Y": coreset_vectors[:, 1],
            "weights": coreset_weights,
        }
    )
    coreset_df["Name"] = [chr(i + 65) for i in coreset_df.index]

    return coreset_df


def get_VQA_cost(
    coreset_df,
    circuit_depth,
    max_iterations,
    max_shots,
    max_cutoff_threshold,
    coreset_to_graph_metric,
    get_K2_Hamiltonian,
    optimizer,
    normalize_vectors,
    optimizer_function,
    create_circuit,
    vector_columns=["X", "Y"],
):
    divisive_clustering = DivisiveClusteringVQA(
        circuit_depth=circuit_depth,
        max_iterations=max_iterations,
        max_shots=max_shots,
        threshold_for_max_cut=max_cutoff_threshold,
        create_Hamiltonian=get_K2_Hamiltonian,
        optimizer=optimizer,
        optimizer_function=optimizer_function,
        create_circuit=create_circuit,
        normalize_vectors=normalize_vectors,
        sort_by_descending=True,
        coreset_to_graph_metric=coreset_to_graph_metric,
    )

    hierarchial_clustering_sequence = divisive_clustering.get_divisive_sequence(
        coreset_df, vector_columns=vector_columns
    )

    return sum(
        divisive_clustering.get_divisive_cluster_cost(
            hierarchial_clustering_sequence,
            coreset_df,
            vector_columns=vector_columns,
        )
    )


def get_KMeans_cost(coreset_df, vector_columns):
    divisive_clustering = DivisiveClusteringKMeans()

    hierarchial_clustering_sequence = divisive_clustering.get_divisive_sequence(
        coreset_df, vector_columns=vector_columns
    )

    return sum(
        divisive_clustering.get_divisive_cluster_cost(
            hierarchial_clustering_sequence,
            coreset_df,
            vector_columns=vector_columns,
        )
    )


def get_Random_cost(coreset_df, vector_columns):
    divisive_clustering = DivisiveClusteringRandom()

    hierarchial_clustering_sequence = divisive_clustering.get_divisive_sequence(
        coreset_df, vector_columns=vector_columns
    )

    return sum(
        divisive_clustering.get_divisive_cluster_cost(
            hierarchial_clustering_sequence,
            coreset_df,
            vector_columns=vector_columns,
        )
    )


def get_MaxCut_cost(coreset_df, vector_columns):
    divisive_clustering = DivisiveClusteringMaxCut()

    hierarchial_clustering_sequence = divisive_clustering.get_divisive_sequence(
        coreset_df, vector_columns=vector_columns
    )

    return sum(
        divisive_clustering.get_divisive_cluster_cost(
            hierarchial_clustering_sequence,
            coreset_df,
            vector_columns=vector_columns,
        )
    )


def main(methods, *args, **kwargs):
    coreset_df = create_coreset_df(
        number_of_rows=kwargs["number_of_rows"],
        coreset_size=kwargs["coreset_size"],
        coreset_method=kwargs["coreset_method"],
        number_of_coresets_to_evaluate=kwargs["number_of_coresets_to_evaluate"],
        number_of_sampling_for_centroids=kwargs["number_of_sampling_for_centroids"],
    )

    cost_dict = {}

    for method in methods:
        print(f"Calculating cost for {method}...")
        if method == "QAOA":
            cost_dict[method] = get_VQA_cost(
                coreset_df=coreset_df,
                circuit_depth=kwargs["circuit_depth"],
                max_iterations=kwargs["max_iterations"],
                max_shots=kwargs["max_shots"],
                max_cutoff_threshold=kwargs["max_cutoff_threshold"],
                coreset_to_graph_metric=kwargs["coreset_to_graph_metric"],
                get_K2_Hamiltonian=kwargs["create_Hamiltonian"],
                optimizer=kwargs["optimizer"],
                normalize_vectors=kwargs["normalize_vectors"],
                optimizer_function=kwargs["optimizer_function_QAOA"],
                create_circuit=kwargs["create_circuit_QAOA"],
                vector_columns=kwargs["vector_columns"],
            )

        elif method == "VQE":
            cost_dict[method] = get_VQA_cost(
                coreset_df=coreset_df,
                circuit_depth=kwargs["circuit_depth"],
                max_iterations=kwargs["max_iterations"],
                max_shots=kwargs["max_shots"],
                max_cutoff_threshold=kwargs["max_cutoff_threshold"],
                coreset_to_graph_metric=kwargs["coreset_to_graph_metric"],
                get_K2_Hamiltonian=kwargs["create_Hamiltonian"],
                optimizer=kwargs["optimizer"],
                normalize_vectors=kwargs["normalize_vectors"],
                optimizer_function=kwargs["optimizer_function_VQE"],
                create_circuit=kwargs["create_circuit_VQE"],
                vector_columns=kwargs["vector_columns"],
            )

        elif method == "KMeans":
            cost_dict[method] = get_KMeans_cost(
                coreset_df=coreset_df, vector_columns=kwargs["vector_columns"]
            )

        elif method == "Random":
            cost_dict[method] = get_Random_cost(
                coreset_df=coreset_df, vector_columns=kwargs["vector_columns"]
            )

        elif method == "MaxCut":
            cost_dict[method] = get_MaxCut_cost(
                coreset_df=coreset_df, vector_columns=kwargs["vector_columns"]
            )

        else:
            raise ValueError(f"Method {method} not supported.")

    print(cost_dict)


if __name__ == "__main__":
    cudaq.set_target(target)

    optimizer = cudaq.optimizers.COBYLA()

    main(
        methods=methods,
        number_of_rows=number_of_rows,
        coreset_size=coreset_size,
        coreset_method=coreset_method,
        number_of_coresets_to_evaluate=number_of_coresets_to_evaluate,
        number_of_sampling_for_centroids=number_of_sampling_for_centroids,
        circuit_depth=circuit_depth,
        max_iterations=max_iterations,
        max_shots=max_shots,
        max_cutoff_threshold=max_cutoff_threshold,
        create_Hamiltonian=get_K2_Hamiltonian,
        optimizer=optimizer,
        optimizer_function_QAOA=get_optimizer_for_QAOA,
        create_circuit_QAOA=get_QAOA_circuit,
        optimizer_function_VQE=get_optimizer_for_VQE,
        create_circuit_VQE=get_VQE_circuit,
        normalize_vectors=normalize_vectors,
        sort_by_descending=True,
        coreset_to_graph_metric=coreset_to_graph_metric,
        vector_columns=["X", "Y"],
    )
