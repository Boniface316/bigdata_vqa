import pandas as pd
from divisiveclustering.coresetsUtils import coreset_to_graph
from divisiveclustering.datautils import DataUtils
from divisiveclustering.plotsutils import plot_all_splits

coreset_numbers = 5
qubits = coreset_numbers
centers = 4
depth = 1
colors = ["blue", "green"]
type_of_outcome = "QAOA"

add_data_label = True
save_plot = True

data_utils = DataUtils("/Users/yogi/libraries/Kmeans_NISQ")

cv, cw, data_vec = data_utils.get_files(coreset_numbers, centers)
coreset_points, G, H, weight_matrix, weights = coreset_to_graph(cv, cw, metric="dot")
df = pd.DataFrame(cv, columns=list("XY"))


data_vec = data_utils.get_raw_data(4)


hc = data_utils.load_object(type_of_outcome, coreset_numbers, centers, depth, "hc")


plot_all_splits(
    type_of_outcome,
    coreset_numbers,
    centers,
    depth,
    hc,
    df,
    colors,
    add_data_label,
    save_plot,
)
