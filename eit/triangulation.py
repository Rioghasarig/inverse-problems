from collections import namedtuple

import numpy as np
from scipy.spatial import Delaunay


DiskMesh = namedtuple(
    "DiskMesh",
    ["points", "triangles", "interior_nodes", "boundary_indices", "boundary_edges"],
)


def triangulate_disk(N, K):
    """Build a Delaunay triangulation of the unit disk.

    Points are placed on ``N`` concentric rings plus a single point at the
    origin. Ring ``i`` (for ``i = 1 ... N``) has radius ``i / N`` and carries
    ``K * i`` evenly spaced points, so the number of points grows with the
    radius and the spacing stays roughly uniform across the disk.

    Parameters
    ----------
    N : int
        Number of concentric rings (the outermost has radius 1).
    K : int
        Base number of points per ring; ring ``i`` gets ``K * i`` points.

    Returns
    -------
    points : numpy.ndarray
        Array of shape ``(2, P)`` where row 0 holds the x coordinates and
        row 1 holds the y coordinates of the ``P`` mesh vertices. The first
        column is the origin ``(0, 0)``.
    triangles : numpy.ndarray
        Array of shape ``(3, T)`` of vertex indices into ``points``, one
        column per triangle.
    interior_nodes : numpy.ndarray
        Indices into ``points`` of every vertex that is *not* on the boundary,
        i.e. the origin and all rings except the outermost. These are the
        unknowns in a Dirichlet problem.
    boundary_indices : numpy.ndarray
        Indices into ``points`` of the boundary vertices, i.e. the points on
        the outermost ring (radius 1), ordered counter-clockwise.
    boundary_edges : numpy.ndarray
        Array of shape ``(2, K * N)`` whose columns each hold the two
        ``points`` indices of a boundary edge connecting consecutive boundary
        vertices around the circle (the last edge wraps back to the first).

    All arrays are returned packaged in a :class:`DiskMesh` named tuple.
    """

    xs = [np.array([0.0])]
    ys = [np.array([0.0])]
    for i in range(1, N + 1):
        r = i / N
        theta = 2 * np.pi * np.arange(K*i) / (K*i)
        xs.append(r * np.cos(theta))
        ys.append(r * np.sin(theta))

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    points = np.vstack((x, y))

    triangles = Delaunay(points.T).simplices.T

    # The outermost ring is the last K * N points appended above; every earlier
    # point (the origin and the inner rings) is therefore interior.
    n_boundary = K * N
    interior_nodes = np.arange(points.shape[1] - n_boundary)
    boundary_indices = np.arange(points.shape[1] - n_boundary, points.shape[1])
    boundary_edges = np.vstack(
        (boundary_indices, np.roll(boundary_indices, -1))
    )

    return DiskMesh(
        points, triangles, interior_nodes, boundary_indices, boundary_edges
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    mesh = triangulate_disk(10, 5)
    points, triangles = mesh.points, mesh.triangles
    plt.triplot(points[0], points[1], triangles.T, color="0.7", lw=0.8)
    for a, b in mesh.boundary_edges.T:
        plt.plot(points[0, [a, b]], points[1, [a, b]], color="C3", lw=1.5)
    plt.plot(
        points[0, mesh.boundary_indices],
        points[1, mesh.boundary_indices],
        "o", color="C3", ms=3,
    )
    plt.gca().set_aspect("equal")
    plt.title(f"{points.shape[1]} points -> {triangles.shape[1]} triangles")
    plt.savefig("triangulation.png", dpi=150)
    plt.show()
