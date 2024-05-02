from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi


class Voironi_Tessalation:
    def __init__(
        self,
        coreset_df: pd.DataFrame,
        clusters: np.ndarray,
        colors: List[str],
        tesslation_by_cluster: Optional[bool] = False,
    ) -> None:
        coreset_df["cluster"] = clusters

        if tesslation_by_cluster:
            cluster_means = coreset_df.groupby("cluster")[["X", "Y"]].mean()
            coreset_df = cluster_means.reset_index()
            coreset_df["cluster"] = [i for i in range(len(coreset_df))]

        coreset_df["color"] = [colors[i] for i in coreset_df.cluster]

        points = coreset_df[["X", "Y"]].to_numpy()

        self.coreset_df = coreset_df

        self.voronoi = Voronoi(points)

    def voronoi_finite_polygons_2d(self, radius: Optional[float] = None) -> Tuple[List, np.ndarray]:
        """
        Creates the Voronoi regions and vertices for 2D data.

        Args:
            radius (Optional[None]): The radius from the data points to create the Voronoi regions. Defaults to None.

        Returns:
            Tuple: The regions and vertices.
        """

        if self.voronoi.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = self.voronoi.vertices.tolist()

        center = self.voronoi.points.mean(axis=0)
        if radius is None:
            radius = self.voronoi.points.ptp().max()

        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(self.voronoi.ridge_points, self.voronoi.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        for p1, region in enumerate(self.voronoi.point_region):
            vertices = self.voronoi.regions[region]

            if all(v >= 0 for v in vertices):
                new_regions.append(vertices)
                continue

            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    continue

                t = self.voronoi.points[p2] - self.voronoi.points[p1]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])

                midpoint = self.voronoi.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n

                far_point = self.voronoi.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)

            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)

    def plot_voironi(
        self,
        plot_title: Optional[str] = "Voronoi Tessalation",
        show_annotation: bool = False,
        show_scatters: bool = False,
    ):
        regions, vertices = self.voronoi_finite_polygons_2d()
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.tight_layout(pad=10)

        for j, region in enumerate(regions):
            polygon = vertices[region]
            color = self.coreset_df.color[j]
            breakpoint()
            plt.fill(*zip(*polygon), alpha=0.4, color=color, linewidth=0)
            if show_annotation:
                plt.annotate(
                    self.coreset_df.Name[j],
                    (self.coreset_df.X[j] + 0.2, self.coreset_df.Y[j]),
                    fontsize=10,
                )

        if show_scatters:
            plt.plot(self.coreset_df.X, self.coreset_df.Y, "ko")

        plt.xlim(min(self.coreset_df.X) - 1, max(self.coreset_df.X) + 1)
        plt.ylim(min(self.coreset_df.Y) - 1, max(self.coreset_df.Y) + 1)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(plot_title)
        plt.show()
