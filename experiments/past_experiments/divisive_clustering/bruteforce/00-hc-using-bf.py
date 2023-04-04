import pandas as pd

from divisiveclustering.bruteforceutils import create_clusters
from divisiveclustering.coresetsUtils import gen_coreset_graph
from divisiveclustering.datautils import DataUtils
from divisiveclustering.helpers import add_children_to_hc, get_centroid_dist_df

coreset_numbers = 5
qubits = coreset_numbers
centers = 4
depth = 1
hc = []
type = "BF"
i = 0
single_clusters = 0

data_utils = DataUtils("/Users/yogi/libraries/Kmeans_NISQ")


cv, cw, data_vec = data_utils.get_files(coreset_numbers, 4)
print(cw)
coreset_points, G, H, weight_matrix, weights = gen_coreset_graph(
    coreset_vectors=cv, coreset_weights=cw
)

index_vals = [i for i in range(coreset_numbers)]
df = pd.DataFrame(cv, columns=list("XY"))
df["Name"] = [chr(i + 65) for i in df.index]

while single_clusters < len(index_vals):

    if i < 1:
        hc = []
        hc.append(index_vals)

    if len(hc[i]) == 1:
        single_clusters += 1
        i += 1
        print("singleton cluster added")
    else:
        sub_index_vals = hc[i]
        sub_df = df.iloc[sub_index_vals]

        sub_df["clusters"], _ = create_clusters(
            "maxcut", sub_df, len(sub_index_vals), cw, cv, sub_index_vals
        )

        hc = add_children_to_hc(sub_df, hc)

        i += 1

dist_df = get_centroid_dist_df(hc, df)
data_utils.save_object(type, coreset_numbers, centers, depth, hc, dist_df, {})
