import pandas as pd
from divisiveclustering.coresetsUtils import coreset_to_graph
from divisiveclustering.datautils import DataUtils
from divisiveclustering.plotsutils import plot_voironi, voronoi_finite_polygons_2d
from scipy.spatial import Voronoi

coreset_numbers = 5
centers = 4

data_utils = DataUtils()

cv, cw, data_vec = data_utils.get_files(coreset_numbers, centers)

df = pd.DataFrame(cv, columns=list("XY"))

cluster_color_dict = data_utils.load_cluster_colors("BF")

points = df.drop(columns=["Name", "cluster"]).to_numpy()
vor = Voronoi(points)
regions, vertices = voronoi_finite_polygons_2d(vor)
plot_voironi(
    df, points, cluster_color_dict, vertices, regions, type_of_plot=type, save_plot=True
)
