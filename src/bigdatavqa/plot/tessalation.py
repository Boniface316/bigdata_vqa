from typing import List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi


class Voironi_Tessalation:
    def _voronoi_finite_polygons_2d(
        self, radius: Optional[float] = None
    ) -> Tuple[List, np.ndarray]:
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
        for (p1, p2), (v1, v2) in zip(
            self.voronoi.ridge_points, self.voronoi.ridge_vertices
        ):
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
        clusters: np.ndarray,
        colors: List[str],
        plot_title: str = "Voronoi Tessalation",
        show_annotation: bool = False,
        show_scatters: bool = False,
    ):
        """

        Plots the Voronoi tessalation.

        Args:
            clusters (np.ndarray): The clusters.
            colors (List[str]): The colors for the clusters.
            plot_title (str): The title of the plot. Defaults to "Voronoi Tessalation".
            show_annotation (bool): Whether to show the annotations. Defaults to False.
            show_scatters (bool): Whether to show the scatters. Defaults to False.

        """
        coreset_df = self.full_coreset_df.copy()
        coreset_df["cluster"] = clusters
        coreset_df["color"] = [colors[i] for i in coreset_df.cluster]

        self.voronoi = Voronoi(coreset_df[self.vector_columns].to_numpy())
        regions, vertices = self._voronoi_finite_polygons_2d()
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.tight_layout(pad=10)

        for j, region in enumerate(regions):
            polygon = vertices[region]
            color = coreset_df.color[j]
            breakpoint()
            plt.fill(*zip(*polygon), alpha=0.4, color=color, linewidth=0)
            if show_annotation:
                plt.annotate(
                    coreset_df.Name[j],
                    (coreset_df.X[j] + 0.2, coreset_df.Y[j]),
                    fontsize=10,
                )

        if show_scatters:
            plt.plot(coreset_df.X, coreset_df.Y, "ko")

        plt.xlim(min(coreset_df.X) - 1, max(coreset_df.X) + 1)
        plt.ylim(min(coreset_df.Y) - 1, max(coreset_df.Y) + 1)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(plot_title)
        plt.show()
