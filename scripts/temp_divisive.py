import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from bigdatavqa.coreset import Coreset
from bigdatavqa.datautils import DataUtils
from bigdatavqa.divisiveclustering import (
    Dendrogram,
    DivisiveClusteringKMeans,
    DivisiveClusteringMaxCut,
    DivisiveClusteringRandom,
    DivisiveClusteringVQE,
    get_divisive_sequence,
)

number_of_qubits = 10
circuit_depth = 1
max_shots = 1000
max_iterations = 100
data_location = "data"
number_of_coresets_to_evaluate = 1
number_of_sampling_for_centroids = 2
threshold_for_max_cut = 0.2

data_utils = DataUtils(data_location)
raw_data = data_utils.load_dataset()

coreset = Coreset(
    raw_data,
    number_of_sampling_for_centroids,
    number_of_qubits,
    number_of_coresets_to_evaluate,
)
coreset_vectors, coreset_weights = coreset.get_best_coresets()


coreset_df = pd.DataFrame(coreset_vectors, columns=list("XY"))
coreset_df["weights"] = coreset_weights
coreset_df["Name"] = [chr(i + 65) for i in coreset_df.index]
coreset_df


# divisive_clustering_function = DivisiveClusteringVQE(
#     circuit_depth=circuit_depth,
#     max_iterations=max_iterations,
#     max_shots=max_shots,
#     threshold_for_max_cut=threshold_for_max_cut,
# )

# hierrachial_sequence = get_divisive_sequence(coreset_df, divisive_clustering_function)

# print(hierrachial_sequence)

# divisive_clustering_function = DivisiveClusteringMaxCut()

# hierrachial_sequence = get_divisive_sequence(coreset_df, divisive_clustering_function)

# print(hierrachial_sequence)

# divisive_clustering_function = DivisiveClusteringRandom()

# hierrachial_sequence = get_divisive_sequence(coreset_df, divisive_clustering_function)

# print(hierrachial_sequence)

divisive_clustering_function = DivisiveClusteringKMeans()

hierrachial_sequence = get_divisive_sequence(coreset_df, divisive_clustering_function)

cost = divisive_clustering_function.get_divisive_cluster_cost(
    hierrachial_sequence, coreset_df
)
