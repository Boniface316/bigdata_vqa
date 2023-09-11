from typing import Dict

import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from orqviz.pca import get_pca, plot_optimization_trajectory_on_pca, plot_pca_landscape

from divisiveclustering.datautils import DataUtils
from divisiveclustering.helpers import find_children


def plot_all_splits(
    type_of_output: str,
    coreset_numbers: int,
    centers: int,
    depth: int,
    hc: list,
    df: pd.DataFrame,
    colors: list,
    add_data_label: bool,
    save_plot: bool,
    folder_path: str = None,
):

    """
    Plot all the splits in the hierarchy

    Args:
        type_of_output: algorithm that created the data
        coreset_numbers: number of coreset
        centers: number of blobs
        depth: depth of circtui
        hc: hierarchial clustering lists
        df: data frame of coreset information
        colors: colors we want to use to classify
        add_data_label: label the data points or not
        save_plot: save the plot or not
        folder_path: where to save the plot

    """
    data_utils = DataUtils(folder_path)
    for parent_posn in range(len(hc)):
        child_lst = find_children(hc, parent_posn)

        if len(child_lst) == 0:
            continue
        else:
            plot_name = data_utils.get_plot_name(
                type_of_output, coreset_numbers, centers, depth, parent_posn
            )
            print(plot_name)
            plot_title = get_plot_title(hc, df, parent_posn)
            draw_plot(
                df,
                child_lst,
                colors,
                add_data_label,
                save_plot,
                plot_name,
                plot_title,
            )


def get_plot_title(hc: list, df: pd.DataFrame, parent_posn: int):
    """
    Get the title of a plot

    Args:
        hc (list): Hierarchy of the data
        df (pd.DataFrame): dataframe of coresets
        parent_posn (int): position of the parent

    Returns:
        _type_: _description_
    """
    df["Name"] = [chr(i + 65) for i in df.index]
    data_point_name = df.loc[hc[parent_posn]]["Name"]

    txt_list = [i for i in data_point_name]

    plot_title = "Clustering the points " + ",".join(txt_list)

    return plot_title


def draw_plot(
    df_plot: pd.DataFrame,
    children_lst: list,
    colors: list,
    add_data_label: bool,
    save_plot: bool,
    plot_name: str,
    plot_title: str,
):

    fig, ax = plt.subplots(1)
    # print(children_lst)

    for i in range(2):

        df_plot_child = df_plot.loc[children_lst[i]]
        x = df_plot_child.to_numpy()[:, 0]
        y = df_plot_child.to_numpy()[:, 1]

        plt.scatter(x, y, s=50, c=colors[i])

        if add_data_label:
            txt_list = get_data_label(df_plot, children_lst, i)
            for j, txt in enumerate(txt_list):
                ax.annotate(txt, (x[j] + 0.1, y[j] + 0.1))

            plt.xlim([min(df_plot.X) - 1, max(df_plot.X) + 1])
            plt.ylim([min(df_plot.Y) - 1, max(df_plot.Y) + 1])

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(plot_title, fontdict={"fontsize": 10})

    if save_plot:
        plt.savefig(plot_name, facecolor="white", transparent=False)


def get_data_label(df: pd.DataFrame, children_lst: list, i: int):

    """
    Get the label for the data

    Args:
        df: data frame of coreset vectors
        children_lst: list of children
        i: iteration number

    Returns:
        _type_: _description_
    """
    data_point_name = df.loc[children_lst[i]]["Name"]

    txt_list = [i for i in data_point_name]

    return txt_list


def find_longest_x(dendo: pd.DataFrame):

    """
    Find the longest branch in the dendrogram

    Args:
        dendo: dendrogram dataframe

    Returns:
        longest branch in the dendrogram
    """
    x1_list = []

    for a in dendo.X1:
        x1_list.append(a[1])

    maxX1 = max(x1_list)

    x2_list = []

    for a in dendo.X2:
        x2_list.append(a[1])

    maxX2 = max(x2_list)

    longest_x = max(maxX1, maxX2)

    return longest_x


def plot_dendogram(dendo: pd.DataFrame):
    """
    Plot the dendrogram

    Args:
        dendo: Dendrogram dataframe
    """
    longest_x = find_longest_x(dendo)
    fig, ax = plt.subplots(figsize=(20, 10))

    for i in range(len(dendo)):
        x1, y1, x2, y2, vx, vy, _, _, Parent, Singleton = dendo.iloc[i]
        if Singleton:
            plt.plot(x1, y1, color="black", marker=" ")
            ax.annotate(Parent, (longest_x + 0.2, y1[0]), fontsize=20)
        else:
            plt.plot(x1, y1, x2, y2, vx, vy, color="black", marker=" ")
    ax.set_yticklabels([])


def plot_clustered_dendogram(
    dendo: pd.DataFrame, dendo_coloring_dict: Dict, cluster_color_dict: Dict
):

    """
    Plot the clustered dendrogram

    Args:
        dendo: dataframe of dendrogram data
        dendo_coloring_dict: dictionary of dendogram coloring
        cluster_color_dict: dictionary of coloring details
    """
    fig, ax = plt.subplots(figsize=(20, 10))

    red_box = dendo_coloring_dict["red"]
    blue_box = dendo_coloring_dict["blue"]
    orange_box = dendo_coloring_dict["orange"]
    purple_box = dendo_coloring_dict["purple"]
    longest_x = dendo_coloring_dict["longest_x"]

    vline = red_box[0]
    plot_title = "Creating 4 clusters by setting the height intercept at " + str(vline)

    for i in range(len(dendo)):
        x1, y1, x2, y2, vx, vy, _, _, Parent, Singleton = dendo.iloc[i]
        if Singleton:
            plt.plot(x1, y1, color="black", marker=" ")
            text_color = cluster_color_dict[Parent]
            ax.annotate(Parent, (longest_x + 0.2, y1[0]), fontsize=25, color=text_color)
        else:
            plt.plot(x1, y1, x2, y2, vx, vy, color="black", marker=" ")
    ax.set_yticklabels([])
    ax.add_patch(
        Rectangle(
            (blue_box[0], blue_box[1]),
            blue_box[2],
            blue_box[3],
            alpha=0.2,
            color="blue",
        )
    )
    ax.add_patch(
        Rectangle(
            (purple_box[0], purple_box[1]),
            purple_box[2],
            purple_box[3],
            alpha=0.2,
            color="purple",
        )
    )
    ax.add_patch(
        Rectangle(
            (orange_box[0], orange_box[1]),
            orange_box[2],
            orange_box[3],
            alpha=0.2,
            color="orange",
        )
    )
    ax.add_patch(
        Rectangle(
            (red_box[0], red_box[1]), red_box[2], red_box[3], alpha=0.2, color="red"
        )
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.axvline(x=vline, color="green", linestyle="--")
    plt.xlabel("Height", fontsize=30)
    plt.xticks(fontsize=30)
    plt.suptitle("Cluster Dendrogram", fontsize=24)
    plt.title(plot_title)


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def plot_voironi(
    df,
    points,
    color_cluster_dict,
    vertices,
    regions,
    type_of_plot,
    data_vec=None,
    add_points=False,
    size=20,
    save_plot=False,
):

    df["Name"] = [chr(i + 65) for i in df.index]
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.tight_layout(pad=10)

    if add_points:
        plt.scatter(data_vec[:, 0], data_vec[:, 1], alpha=0.5)

    for j, region in enumerate(regions):
        polygon = vertices[region]
        color = color_cluster_dict[df.Name[j]]
        plt.fill(*zip(*polygon), alpha=0.4, color=color, linewidth=0)
        ax.annotate(df.Name[j], (df.X[j] + 0.2, df.Y[j]), fontsize=size)

    plt.plot(points[:, 0], points[:, 1], "ko")
    plt.xlim(min(df.X) - 1, max(df.X) + 1)
    plt.ylim(min(df.Y) - 1, max(df.Y) + 1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("x", fontsize=30)
    plt.ylabel("y", fontsize=30)

    # plt.title("Voronoi diagram of the clusters \n", fontsize = 25)
    if save_plot:
        plt.savefig(
            type_of_plot + "_k_4_voironi.png", facecolor="white", transparent=False
        )


def plot_k4_cluster(
    df: pd.DataFrame,
    cluster_color_dict: Dict,
):

    """
    Plot the scatter plot for k = 4

    Args:
        df: dataframe of coreset vectors
        cluster_color_dict: dictionary of how the points are colored

    """
    df["cluster"] = 0

    for key, value in cluster_color_dict.items():
        df.loc[df["Name"] == key, "cluster"] = cluster_color_dict[key]

    df_groups = df.groupby("cluster")

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.tight_layout(pad=10)

    x = df.X.to_numpy()
    y = df.Y.to_numpy()

    for name, group in df_groups:
        clus_color = group.cluster
        clus_color = clus_color.to_list()
        clus_color = clus_color[0]
        plt.plot(
            group["X"],
            group["Y"],
            marker="o",
            linestyle="",
            label=name,
            markersize=10,
            color=clus_color,
        )

    ax.set_xlim([min(x) - 1, max(x) + 1])
    ax.set_ylim([min(y) - 2, max(y) + 2])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("x", fontsize=30)
    plt.ylabel("y", fontsize=30)


def get_trajectories_collection(cost_params_dict: Dict):
    """
    Get the collection of all tractorires for plots

    Args:
        cost_params_dict: Dictionary of parameters about the cost

    Returns:
        Trajectories
    """
    trajectories_list = cost_params_dict["trajectories"]
    param_size = len(cost_params_dict["params"][0])

    trajectories_collection = np.zeros(param_size).reshape(-1, param_size)

    for trajectory in trajectories_list:
        trajectories_collection = np.append(trajectories_collection, trajectory, axis=0)

    trajectories_collection = np.delete(trajectories_collection, 0, axis=0)

    return trajectories_collection, trajectories_list


def plot_optimization_trajectory(pca_result, cost_params_dict, colors=None):
    """
    Plot the trajectory of the optimization path

    Args:
        pca_result: PCA results created by Orqviz
        cost_params_dict: Cost parameters dictionary
        colors: Colors selected for each trajectory
    """
    trajectories_collection, trajectories_list = get_trajectories_collection(
        cost_params_dict
    )

    pca = get_pca(trajectories_collection)
    if colors is None:
        colors = ["lightsteelblue", "green", "red", "white"]

    fig, ax = plt.subplots()
    plot_pca_landscape(pca_result, pca, fig=fig, ax=ax)
    for i, trajectory in enumerate(trajectories_list):
        plot_optimization_trajectory_on_pca(
            trajectory, pca, ax=ax, label="Optimization Trajectory", color=colors[i]
        )
