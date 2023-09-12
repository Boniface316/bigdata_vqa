import numpy as np
import pandas as pd
from divisiveclustering.coresetsUtils import coreset_to_graph
from divisiveclustering.datautils import DataUtils
from divisiveclustering.helpers import (
    create_empty_dendo_df,
    extend_singletons,
    find_children,
    find_parent_clus_posn,
    find_parent_loc,
    get_centroid_dist_df,
    get_cluster_num_dict,
    get_cluster_position,
    get_dendo_xy,
    get_parent_location,
    get_xy_val,
)
from divisiveclustering.plotsutils import (
    find_longest_x,
    plot_clustered_dendogram,
    plot_dendogram,
)
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.pyplot import figure

coreset_numbers = 5
qubits = coreset_numbers
centers = 4
depth = 1

data_utils = DataUtils()

cv, cw, data_vec = data_utils.get_files(coreset_numbers, centers)

coreset_points, G, H, weight_matrix, weights = coreset_to_graph(cv, cw, metric="dot")

df = pd.DataFrame(cv, columns=list("XY"))

df["Name"] = [chr(i + 65) for i in df.index]

hc = data_utils.load_object("BF", coreset_numbers, centers, depth, "hc")

dist_df = get_centroid_dist_df(hc, df)

cluster_dict = get_cluster_num_dict(hc, dist_df)

# dendo_df = pd.DataFrame(columns=['X1', 'Y1', 'X2', 'Y2', 'VX' , 'VY',"Cluster1", "Cluster2",  "Parent", "Singleton"], index=range(len(hc)))

dendo_df = create_empty_dendo_df(hc)


################# Start dendrogram dataframe creation ############
# This is done manually
# Use Jupyternotebook to create dendrogram
# Start with current_posn at 0 and end at the len(hc)
current_posn = 0


cluster_position = get_cluster_position(cluster_dict, hc, current_posn)

parent_location = get_parent_location(hc, current_posn)

x_start, y_val = get_xy_val(parent_location, current_posn, dendo_df, cluster_position)

print(x_start)
print(y_val)

print(find_children(hc, current_posn, cluster_dict))

# use x_start, y_val and the children to adjust the buffer. The buffer decides how far the branches are from each other.
# Ensure that children with few number of points are closer to the origin
# This enasure that we can expand on the other side


buffer = 100

xy_lst = get_dendo_xy(
    cluster_dict, dist_df, current_posn, hc, x_start, [y_val - buffer, y_val + buffer]
)


for i, col_name in enumerate(dendo_df.columns):
    dendo_df.iloc[current_posn][col_name] = xy_lst[i]


############### Stop creating dendrogram data frame #############


longest_x = find_longest_x(dendo_df)

singleton_posns = dendo_df.index[dendo_df.Singleton]

dendo_df = extend_singletons(singleton_posns, cluster_dict, hc, dendo_df, longest_x)


# fig, ax = plt.subplots(figsize=(20, 10))

# for i in range(len(dendo_a)):
#     x1, y1, x2, y2, vx, vy, _, _, Parent, Singleton = dendo_a.iloc[i]
#     if Singleton:
#         plt.plot(x1, y1, color="black", marker=" ")
#         ax.annotate(Parent, (longest_x + 0.2, y1[0]), fontsize=20)
#     else:
#         plt.plot(x1, y1, x2, y2, vx, vy, color="black", marker=" ")
# ax.set_yticklabels([])
# plt.axvline(x=3, color='green', linestyle='--')
# plt.savefig("vqe_dendrogram.png", facecolor='white', transparent=False)

data_utils.save_dendo_data(dendo_df, "BF")


# Plot the dendrogram first and find the cut off pointt
# Then assign a color for each node
# This is done manually

plot_dendogram(dendo_df)

cluster_color_dict = {
    "A": "blue",
    "B": "purple",
    "C": "red",
    "D": "blue",
    "E": "purple",
    "F": "orange",
    "G": "red",
    "H": "red",
    "I": "orange",
    "J": "purple",
    "K": "red",
    "L": "blue",
}

# Use the dendrogram and the dendrogramn_df to come up with the values below
cut_off_line = 7.5

blue_rect_x1 = cut_off_line
blue_rect_y1 = -875
blue_width = longest_x - cut_off_line
blue_height = 875 - 450

red_rect_x1 = cut_off_line
red_rect_y1 = 350
red_width = longest_x - cut_off_line
red_height = 1000 - 300

orange_rect_x1 = cut_off_line
orange_rect_y1 = 10
orange_width = longest_x - cut_off_line
orange_height = 200

purple_rect_x1 = cut_off_line
purple_rect_y1 = -425
purple_width = longest_x - cut_off_line
purple_height = 425 - 20

dendo_coloring_dict = {
    "red": [red_rect_x1, red_rect_y1, red_width, red_height],
    "orange": [orange_rect_x1, orange_rect_y1, orange_width, orange_height],
    "purple": [purple_rect_x1, purple_rect_y1, purple_width, purple_height],
    "blue": [blue_rect_x1, blue_rect_y1, blue_width, blue_height],
    "longest_x": longest_x,
}


plot_clustered_dendogram(dendo_df, dendo_coloring_dict, cluster_color_dict)
