import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.pyplot import figure

from divisiveclustering.coresetsUtils import gen_coreset_graph
from divisiveclustering.datautils import DataUtils
from divisiveclustering.helpers import (
    create_empty_dendo_df,
    extend_singletons,
    find_children,
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

coreset_numbers = 5
qubits = coreset_numbers
centers = 4
depth = 1

data_utils = DataUtils()

cv, cw, data_vec = data_utils.get_files(coreset_numbers, centers)

coreset_points, G, H, weight_matrix, weights = gen_coreset_graph(cv, cw, metric="dot")

df = pd.DataFrame(cv, columns=list("XY"))

df["Name"] = [chr(i + 65) for i in df.index]

hc = data_utils.load_object("VQE", coreset_numbers, centers, depth, "hc")

dist_df = get_centroid_dist_df(hc, df)

cluster_dict = get_cluster_num_dict(hc, dist_df)

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

data_utils.save_dendo_data(dendo_df, "VQE")


# Plot the dendrogram first and find the cut off pointt
# Then assign a color for each node
# This is done manually

plot_dendogram(dendo_df)

cluster_color_dict = {
    "A": "blue",
    "B": "purple",
    "C": "blue",
    "D": "orange",
    "E": "red",
    "F": "red",
    "G": "red",
    "H": "red",
    "I": "blue",
    "J": "blue",
    "K": "blue",
    "L": "blue",
}

# Use the dendrogram and the dendrogramn_df to come up with the values below
cut_off_line = 1.8

red_rect_x1 = cut_off_line
red_rect_y1 = 375
red_width = longest_x - cut_off_line
red_height = 1000 - 325

orange_rect_x1 = cut_off_line
orange_rect_y1 = 10
orange_width = longest_x - cut_off_line
orange_height = 100

purple_rect_x1 = cut_off_line
purple_rect_y1 = -100
purple_width = longest_x - cut_off_line
purple_height = 80

blue_rect_x1 = cut_off_line
blue_rect_y1 = -1100
blue_width = longest_x - cut_off_line
blue_height = 950

dendo_coloring_dict = {
    "red": [red_rect_x1, red_rect_y1, red_width, red_height],
    "orange": [orange_rect_x1, orange_rect_y1, orange_width, orange_height],
    "purple": [purple_rect_x1, purple_rect_y1, purple_width, purple_height],
    "blue": [blue_rect_x1, blue_rect_y1, blue_width, blue_height],
    "longest_x": longest_x,
}


plot_clustered_dendogram(dendo_df, dendo_coloring_dict, cluster_color_dict)