import re
from typing import Dict, Sequence, Union

import numpy as np
import pandas as pd


def add_children_to_hc(df: pd.DataFrame, hc: list):
    """
    Add the children to the hierachy structure

    Args:
        df: dataframe
        hc: hierachy structure

    Returns:
        updated list with children
    """
    for j in range(2):
        idx = list(df[df["clusters"] == j].index)
        if len(idx) > 0:
            hc.append(idx)

    return hc


def get_centroid_dist_df(hc, df):
    """
    Get the distance of centroids

    Args:
        df: dataframe
        hc: hierachy structure

    Returns:
        distance matrix
    """

    centroids = get_centroid_coords(hc, df)
    d = get_centroid_dist(centroids)

    d = rename_cols_rows(d, df)

    return d


def get_centroid_coords(hc, df):
    """
    Get centroid coordination

    Args:
        df: dataframe
        hc: hierachy structure

    Returns:
        Centroids coordination
    """
    res = []
    [res.append(x) for x in hc if x not in res]

    centroids = {}

    for posn in res:
        df_temp = df.iloc[posn]
        if "Name" in df_temp.columns:
            df_temp = df_temp.drop(columns=["Name"])
        elif "cluster" in df_temp.columns:
            df_temp = df_temp.drop(columns=["cluster"])

        centroid = np.mean(df_temp.to_numpy(), axis=0)

        centroids.update({str(posn): centroid})

    return centroids


def get_centroid_dist(centroids):
    """
    Centroid distance

    Args:
        centroids: centoids data

    Returns:
        Distance between centroid and data point
    """

    d = create_df_for_centroids(centroids)

    for idx in d.index:
        for col in d.columns:
            a = centroids[idx]
            b = centroids[col]
            d.loc[idx, col] = np.linalg.norm(a - b)
    return d


def rename_cols_rows(d: pd.DataFrame, df: pd.DataFrame):
    """
    Rename columns and rows

    Args:
        d: centroids data frame
        df: dataframe

    Returns:
        _type_: _description_
    """

    cluster_names = create_singleton_cluster_dict(df)

    col = d.columns[0]

    col_names = []

    for col in d.columns:
        a = re.sub(r"[\[\]]", "", col)
        b = ""
        for chr_val in a.split(","):
            chr_val = chr_val.replace(" ", "")
            b += cluster_names[str(chr_val)]
        col_names.append(b)

    d.columns = col_names
    d.index = col_names

    return d


def create_df_for_centroids(centroids: pd.DataFrame):
    """
    Create an empty datafram to store centroid details
    Args:
        centroids: centroids data

    Returns:
        _type_: _description_
    """

    d = pd.DataFrame(0, index=np.arange(len(centroids)), columns=centroids.keys())

    d.index = centroids.keys()

    return d


def create_singleton_cluster_dict(df: pd.DataFrame):
    """
    Create a dictionary of all singleton clusters

    Args:
        df: dataframe of entire data sets

    Returns:
        singleton clusters name
    """
    cluster_names = {}

    for idx in df.index:
        cluster_names.update({str(idx): df.Name[idx]})

    return cluster_names


def find_children(hc: list, parent_posn: int, cluster_dict: Dict = None):
    """

    Find children of a given parent
    Args:
        hc: Hierarchy structure
        parent_posn: Position of a parent

    Returns:
        Name os children
    """
    index_vals = hc[parent_posn]

    if len(index_vals) == 1:
        return []

    child_posn = []
    child_list = []
    children_list = []

    for instance in index_vals:
        # current_posn = parent_posn
        for i in range(parent_posn + 1, len(hc)):
            idx_val = hc[i]

            if instance in idx_val:
                if instance not in child_list:
                    # child_list.append(idx_val)
                    child_posn.append(i)
                    break

    children_posn = list(dict.fromkeys(child_posn))

    if cluster_dict is None:
        for child in children_posn:
            child_list.append(hc[child])

        return child_list

    else:

        for child in children_posn:
            children_list.append(hc[child])

        if children_list:

            child1_str = str(children_list[0])
            child2_str = str(children_list[1])

            child1_name = cluster_dict[child1_str]
            child2_name = cluster_dict[child2_str]

            return [child1_name, child2_name]
        else:
            return children_list


def get_cluster_num_dict(hc: list, dist_df: pd.DataFrame):
    """
    create a dictionary with cluster number and cluster name

    Args:
        hc: hierarchy
        dist_df: distance dataframe

    Returns:
        dictionary with cluster name and numbers
    """
    dict = {}
    colnames = dist_df.columns

    for i, cluster in enumerate(hc):
        clus_name = colnames[i]
        dict.update({str(cluster): clus_name})

    return dict


def create_empty_dendo_df(hc: list):
    """
    Create an empty dataframe to store dendrogram data

    Args:
        hc: hierarchial data

    Returns:
        empty data frame
    """
    return pd.DataFrame(
        columns=[
            "X1",
            "Y1",
            "X2",
            "Y2",
            "VX",
            "VY",
            "Cluster1",
            "Cluster2",
            "Parent",
            "Singleton",
        ],
        index=range(len(hc)),
    )


def find_parent_clus_posn(parent_cluster: int, dendo_df: pd.DataFrame):
    """
    Find the cluster position of a parent on a dendrogram.

    Args:
        parent_cluster (int): number of the parent cluster
        dendo_df (pd.DataFrame): dendrogram dataframe

    Returns:
        Position of the cluster
    """
    clus1 = dendo_df[dendo_df.Cluster1 == parent_cluster]
    clus2 = dendo_df[dendo_df.Cluster2 == parent_cluster]

    clusPosn = None

    if len(clus1) > 0:
        clusPosn = "C1"
    elif len(clus2) > 0:
        clusPosn = "C2"
    else:
        "cant find"

    return clusPosn


def get_xy_val(parent_loc: int, current_posn: int, dendo_df: int, clusPosn: int):
    """
    Find the coordinate in a dendogram for a given parent

    Args:
        parent_loc: Location of a parent in hc
        current_posn: Current position in the hc
        dendo_df: Dendrogram dataframe
        clusPosn: Cluster position in the dendrogram

    Returns:
        Coordinates in the dendrogram
    """
    x_start = 0
    y_val = 0

    if current_posn == 0:
        x_start = 0
        y_val = 0

    else:
        breakpoint()
        if clusPosn == "C1":
            x_start = dendo_df.iloc[parent_loc]["X1"][1]
            y_val = dendo_df.iloc[parent_loc]["Y1"][1]
        else:
            x_start = dendo_df.iloc[parent_loc]["X2"][1]
            y_val = dendo_df.iloc[parent_loc]["Y2"][1]
    return x_start, y_val


def find_parent_loc(parent_hc: list, child_val: int):
    """
    Find the location of a parent

    Args:
        parent_hc: parent hierarchy
        child_val: value of a child

    Returns:
        Location of a parent
    """
    for i, hc_val in enumerate(parent_hc):
        for idx in hc_val:
            if idx == child_val[0]:
                parent_loc = len(parent_hc) - 1 - i
                return parent_loc


def get_dendo_xy(
    cluster_dict: Dict,
    dist_df: pd.DataFrame,
    parent_posn: int,
    hc: list,
    x_start: float,
    height: float,
):

    """
    Get the values on the dendrogram

    Args:
        cluster_dict: dictionary about cluster details
        dist_df: distance matrix
        parent_posn: position of the parent
        hc: hierarchy structure
        x_start: where to start on the x-axis
        heigh: height at which the line should beging

    Returns:
        list of values
    """
    children_lst = find_children(hc, parent_posn, cluster_dict)
    parent_cluster = cluster_dict[str(hc[parent_posn])]
    if children_lst:
        child1_name, child2_name = children_lst

    else:
        return [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            None,
            None,
            parent_cluster,
            True,
        ]

    a = dist_df[[parent_cluster]].loc[child1_name]
    a = float(a)

    b = dist_df[[parent_cluster]].loc[child2_name]
    b = float(b)

    x1, y1 = [x_start, x_start + a], [height[0], height[0]]
    x2, y2 = [x_start, x_start + b], [height[1], height[1]]
    vx, vy = [x_start, x_start], [height[0], height[1]]

    return [x1, y1, x2, y2, vx, vy, child1_name, child2_name, parent_cluster, False]


def get_cluster_position(cluster_dict: Dict, hc: list, current_posn: int):
    """
    Get the position of a cluster

    Args:
        cluster_dict (Dict): Dictionary about the cluster
        hc (list): Hierarchy details
        current_posn (int): Current position in the hierarchy

    Returns:
        cluster position
    """

    current_cluster = cluster_dict[str(hc[current_posn])]

    cluster_position = find_parent_clus_posn(current_cluster, dendo_df)

    return cluster_position


def get_parent_location(hc: list, current_posn: int):
    """
    Find the location of a parent

    Args:
        hc (list): hierarchy of the data
        current_posn (int): curent position in the hc

    Returns:
        Parent's location
    """

    parent_hc = hc[:current_posn]
    parent_hc = parent_hc[::-1]

    child_val = hc[current_posn]
    parent_loc = find_parent_loc(parent_hc, child_val)

    return parent_loc


def extend_singletons(
    singleton_posns: Sequence[list],
    cluster_dict: Dict,
    hc: list,
    dendo_df: pd.DataFrame,
    longest_x: float,
):

    """
    Extends the singleton clusters to the end of the graph

    Args:

        singleton_posn: position of singletons
        cluster_dict: dictionary of clusters
        hc: Hierarchy of the data
        dendo_df: Dendrogram details as dataframe
        longest_x: the longest distance in the x direction


    Returns:
        Dataframe with extended singleton
    """
    for current_posn in singleton_posns:

        parent_cluster = cluster_dict[str(hc[current_posn])]

        clus1 = dendo_df[dendo_df.Cluster1 == parent_cluster]
        clus2 = dendo_df[dendo_df.Cluster2 == parent_cluster]

        if len(clus1) > 0:
            clusPosn = "C1"
        elif len(clus2) > 0:
            clusPosn = "C2"
        else:
            "cant find the cluster"

        parent_loc = get_parent_location(hc, current_posn)

        xy_coords = dendo_df.loc[parent_loc]

        if clusPosn == "C1":
            x1 = [xy_coords[0][1], longest_x]
            y1 = [xy_coords[1][1], xy_coords[1][1]]
        elif clusPosn == "C2":
            x1 = [xy_coords[2][1], longest_x]
            y1 = [xy_coords[3][1], xy_coords[3][1]]

        dendo_df.loc[current_posn]["X1"] = x1
        dendo_df.loc[current_posn]["Y1"] = y1

        return dendo_df


def np_to_bitstring(bitstring: np.ndarray):
    """
    Convert numpy to bitstring

    Args:
        bitstring (int): bitstring

    Returns:
        bitstring as a string
    """
    bitstring = np.array2string(bitstring)
    bitstring = "".join(bitstring.split())
    bitstring = bitstring[1 : len(bitstring) - 1]

    return bitstring
