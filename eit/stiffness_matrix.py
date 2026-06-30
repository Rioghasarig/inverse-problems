from collections import namedtuple

import numpy as np
import scipy.sparse as sp


def local_stiffness(vertices, sigma=1.0):
    """Element stiffness matrix for one linear (P1) triangle.

    Discretizes the bilinear form ``a(u, v) = \\int_T sigma grad(u).grad(v) dA``
    on a single triangle ``T``. The P1 basis functions are linear over the
    triangle, so their gradients are constant and the integral reduces to
    ``sigma * |T| * (grad phi_m . grad phi_n)``.

    Parameters
    ----------
    vertices : array_like, shape (2, 3)
        Coordinates of the triangle's three vertices: row 0 holds the x
        coordinates, row 1 the y coordinates (one column per vertex). This
        matches a column of ``triangulation.triangulate_disk``'s ``points``
        indexed by a column of its ``triangles`` array.
    sigma : float, optional
        Conductivity on the triangle (assumed constant per element).
        Defaults to ``1.0``.

    Returns
    -------
    numpy.ndarray
        Symmetric ``(3, 3)`` local stiffness matrix whose ``(m, n)`` entry is
        ``sigma * |T| * grad(phi_m) . grad(phi_n)``. Its rows (and columns)
        sum to zero, since a constant function has zero gradient.
    """
    vertices = np.asarray(vertices, dtype=float)
    x = vertices[0]
    y = vertices[1]

    # Each P1 (hat) basis function phi_m is the unique linear function that is 1
    # at vertex m and 0 at the other two vertices. On the triangle it has the form
    # phi_m(x, y) = (a_m + beta_m * x + gamma_m * y) / (2|T|). The coefficients
    # beta_m, gamma_m are exactly the entries below; they come from inverting the
    # 3x3 system that pins phi_m to (1, 0, 0) at the three vertices, which by
    # Cramer's rule yields these differences of the *opposite* edge's coordinates.

    # Signed area of the triangle: |T| = 1/2 |(p2-p1) x (p3-p1)| (the cross
    # product of two edge vectors). We keep abs() below since orientation
    # (vertex ordering) must not flip the sign of the stiffness entries.
    area = 0.5 * ((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))

    # Because phi_m is linear, its gradient is the constant vector
    #     grad phi_m = (1 / 2|T|) * [beta_m, gamma_m],
    # where beta_m = y_{m+1} - y_{m+2} and gamma_m = x_{m+2} - x_{m+1}
    # (indices taken cyclically). Geometrically [beta_m, gamma_m] is the inward
    # normal of the edge opposite vertex m, scaled by that edge's length.
    beta = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
    gamma = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])

    # The integrand grad(phi_m).grad(phi_n) is constant over T, so the element
    # integral is just (constant value) * area:
    #   A_mn = sigma * |T| * grad(phi_m).grad(phi_n)
    #        = sigma * |T| * (beta_m*beta_n + gamma_m*gamma_n) / (2|T|)^2
    #        = sigma * (beta_m*beta_n + gamma_m*gamma_n) / (4|T|).
    # The two outer products assemble the beta.beta and gamma.gamma parts for all
    # (m, n) pairs at once.
    return sigma / (4.0 * abs(area)) * (np.outer(beta, beta) + np.outer(gamma, gamma))


def assemble_stiffness(mesh, sigma=1.0):
    """Assemble the global P1 stiffness matrix for ``div(sigma grad u) = 0``.

    Builds the sparse matrix ``A`` with entries
    ``A_ij = \\int_Omega sigma grad(phi_i).grad(phi_j) dA`` by summing the
    per-triangle contributions from :func:`local_stiffness`. ``A`` is symmetric
    positive semi-definite and singular: the constant vector lies in its null
    space (a constant has zero gradient), so it only becomes solvable once
    boundary conditions are imposed (see :func:`dirichlet_system`).

    Parameters
    ----------
    mesh : triangulation.DiskMesh
        Mesh providing ``points`` (shape ``(2, P)``) and ``triangles``
        (shape ``(3, T)``).
    sigma : float or array_like, optional
        Conductivity, assumed constant per element. Either a scalar (broadcast
        to every triangle) or a length-``T`` array, one value per triangle.
        Defaults to ``1.0``.

    Returns
    -------
    scipy.sparse.csr_matrix
        The ``(P, P)`` global stiffness matrix.
    """
    points = mesh.points
    triangles = mesh.triangles
    P = points.shape[1]
    T = triangles.shape[1]

    # Broadcast a scalar conductivity to one value per triangle.
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (T,))

    # Accumulate the 9 entries of each 3x3 local matrix as COO triplets. Building
    # one coo_matrix at the end is far cheaper than indexing a matrix in the loop,
    # and COO sums duplicate (row, col) pairs -- exactly the FEM assembly rule.
    rows = np.empty(9 * T, dtype=np.intp)
    cols = np.empty(9 * T, dtype=np.intp)
    vals = np.empty(9 * T, dtype=float)

    for t in range(T):
        g = triangles[:, t]                      # global indices of the 3 vertices
        A_loc = local_stiffness(points[:, g], sigma[t])

        sl = slice(9 * t, 9 * t + 9)
        # Scatter A_loc[m, n] -> A[g[m], g[n]]: repeat rows, tile cols.
        rows[sl] = np.repeat(g, 3)
        cols[sl] = np.tile(g, 3)
        vals[sl] = A_loc.ravel()

    return sp.coo_matrix((vals, (rows, cols)), shape=(P, P)).tocsr()


DirichletSystem = namedtuple(
    "DirichletSystem",
    ["A_II", "rhs", "interior_indices", "boundary_indices", "boundary_values"],
)


def dirichlet_system(mesh, sigma, g):
    """Reduce the stiffness matrix to the interior system for a Dirichlet problem.

    Given ``div(sigma grad u) = 0`` with ``u = g`` prescribed on the boundary,
    partition the assembled system ``A U = 0`` into interior (``I``) and
    boundary (``B``) nodes. With the boundary values ``U_B = g`` known, the
    first block row becomes the solvable system ::

        A_II U_I = -A_IB U_B = rhs

    where ``A_II`` is symmetric positive definite. Solving it (e.g. with
    ``scipy.sparse.linalg.spsolve``) gives the interior nodal values; combining
    with ``U_B`` at ``boundary_indices`` recovers the full solution.

    Parameters
    ----------
    mesh : triangulation.DiskMesh
        Mesh providing ``points``, ``triangles``, ``interior_nodes`` and
        ``boundary_indices``.
    sigma : float or array_like
        Conductivity passed through to :func:`assemble_stiffness`.
    g : array_like or callable
        Boundary data. Either a length-``B`` array giving the prescribed value
        at each node in ``mesh.boundary_indices`` (same order), or a callable
        ``g(x, y)`` evaluated at the boundary node coordinates.

    Returns
    -------
    DirichletSystem
        Named tuple ``(A_II, rhs, interior_indices, boundary_indices,
        boundary_values)``: the reduced ``(n_I, n_I)`` matrix and ``rhs``
        vector, plus the index arrays and boundary values needed to map the
        interior solution back to the full ``P``-vector.
    """
    A = assemble_stiffness(mesh, sigma)

    B = np.asarray(mesh.boundary_indices, dtype=np.intp)
    I = np.asarray(mesh.interior_nodes, dtype=np.intp)

    # Evaluate the prescribed boundary values U_B at the boundary nodes.
    if callable(g):
        U_B = np.asarray(g(mesh.points[0, B], mesh.points[1, B]), dtype=float)
    else:
        U_B = np.asarray(g, dtype=float)
    if U_B.shape != B.shape:
        raise ValueError(
            f"boundary data has length {U_B.shape}, expected {B.shape}"
        )

    # Block-partition A (CSR row-slice then column-slice) and move the known
    # boundary contribution to the right-hand side.
    A_II = A[I][:, I]
    A_IB = A[I][:, B]
    rhs = -(A_IB @ U_B)

    return DirichletSystem(A_II, rhs, I, B, U_B)
