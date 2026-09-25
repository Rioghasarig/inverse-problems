import numpy as np


def triangle_centroids(mesh):
    """Centroid coordinates of every triangle in a mesh.

    Parameters
    ----------
    mesh : triangulation.DiskMesh
        Mesh providing ``points`` (shape ``(2, P)``) and ``triangles``
        (shape ``(3, T)``).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(2, T)``: row 0 holds the x coordinate and row 1 the
        y coordinate of each triangle's centroid (the mean of its 3 vertices).
    """
    # points[:, triangles] has shape (2, 3, T); average over the 3 vertices.
    return mesh.points[:, mesh.triangles].mean(axis=1)


def gaussian_bumps(specs, sigma_bg=1.0, sigma_min=1e-3):
    """Build a conductivity field made of several Gaussian bumps on a background.

    The returned field is ::

        sigma(x, y) = sigma_bg + sum_k amp_k * exp(-r_k^2 / width_k^2)

    where ``r_k`` is the distance from ``(x, y)`` to bump ``k``'s center. Bumps
    superpose (overlapping ones add), amplitudes may be negative to model
    low-conductivity regions, and the result is clipped to ``sigma_min`` so the
    conductivity stays strictly positive (required for the SPD FEM assembly).

    Parameters
    ----------
    specs : iterable of (center, width, amp)
        One entry per bump, where ``center = (cx, cy)`` is the bump location,
        ``width`` is the Gaussian length scale, and ``amp`` is the peak height
        above (or below, if negative) the background.
    sigma_bg : float, optional
        Constant background conductivity. Defaults to ``1.0``.
    sigma_min : float, optional
        Lower clip applied to the field so it never reaches zero or goes
        negative. Defaults to ``1e-3``.

    Returns
    -------
    callable
        A vectorized ``field(x, y)`` taking array-like coordinates and returning
        the conductivity at each point, with the same shape as ``x``.
    """
    specs = list(specs)

    def field(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        s = np.full(np.broadcast(x, y).shape, float(sigma_bg))
        for (cx, cy), width, amp in specs:
            r2 = (x - cx) ** 2 + (y - cy) ** 2
            s = s + amp * np.exp(-r2 / width ** 2)
        return np.clip(s, sigma_min, None)

    return field


def random_gaussian_bumps(
    n,
    rng=None,
    max_radius=0.8,
    width=(0.1, 0.25),
    amp=(0.5, 2.0),
    sigma_bg=1.0,
    sigma_min=1e-3,
):
    """Convenience builder for ``n`` randomly placed Gaussian bumps.

    Bump centers are drawn uniformly over the disk of radius ``max_radius``
    (using ``sqrt`` of a uniform radius so they spread evenly by area rather than
    clustering at the origin). Keeping ``max_radius < 1`` holds the bumps away
    from the boundary, which is the usual setup for EIT phantoms.

    Parameters
    ----------
    n : int
        Number of bumps.
    rng : int, numpy.random.Generator, or None, optional
        Seed or generator for reproducible phantoms. Passed to
        ``numpy.random.default_rng``.
    max_radius : float, optional
        Maximum distance of a bump center from the origin. Defaults to ``0.8``.
    width : tuple of float, optional
        ``(low, high)`` range the per-bump width is drawn from uniformly.
    amp : tuple of float, optional
        ``(low, high)`` range the per-bump amplitude is drawn from uniformly.
    sigma_bg, sigma_min : float, optional
        Passed through to :func:`gaussian_bumps`.

    Returns
    -------
    callable
        A ``field(x, y)`` as returned by :func:`gaussian_bumps`.
    """
    rng = np.random.default_rng(rng)
    specs = []
    for _ in range(n):
        r = max_radius * np.sqrt(rng.uniform())
        theta = rng.uniform(0.0, 2.0 * np.pi)
        center = (r * np.cos(theta), r * np.sin(theta))
        specs.append((center, rng.uniform(*width), rng.uniform(*amp)))
    return gaussian_bumps(specs, sigma_bg=sigma_bg, sigma_min=sigma_min)


def sample_sigma(mesh, field):
    """Sample a conductivity field at the triangle centroids of a mesh.

    Produces the per-triangle conductivity array consumed by
    ``stiffness_matrix.assemble_stiffness`` / ``dirichlet_system``. Sampling at
    centroids matches P1's piecewise-constant-per-element conductivity model.

    Parameters
    ----------
    mesh : triangulation.DiskMesh
        Mesh providing ``points`` and ``triangles``.
    field : callable
        A ``field(x, y)`` such as those returned by :func:`gaussian_bumps` or
        :func:`random_gaussian_bumps`.

    Returns
    -------
    numpy.ndarray
        Length-``T`` array (one conductivity value per triangle).
    """
    cx, cy = triangle_centroids(mesh)
    return np.asarray(field(cx, cy), dtype=float)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from triangulation import triangulate_disk

    mesh = triangulate_disk(20, 8)

    field = gaussian_bumps(
        [
            ((0.4, 0.0), 0.20, 1.5),    # high-conductivity blob, right
            ((-0.3, 0.3), 0.15, 1.0),   # smaller blob, upper-left
            ((0.0, -0.4), 0.25, -0.6),  # low-conductivity dip, bottom
        ]
    )
    sigma = sample_sigma(mesh, field)
    print(f"{sigma.shape[0]} triangles, sigma in [{sigma.min():.3f}, {sigma.max():.3f}]")

    points, triangles = mesh.points, mesh.triangles
    tpc = plt.tripcolor(
        points[0], points[1], triangles.T, facecolors=sigma, edgecolors="0.7", lw=0.2
    )
    plt.colorbar(tpc, label="conductivity sigma")
    plt.gca().set_aspect("equal")
    plt.title("Gaussian-bump conductivity phantom")
    plt.savefig("conductivity.png", dpi=150)
    plt.show()
