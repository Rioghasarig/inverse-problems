"""The Dirichlet-to-Neumann map of the conductivity equation on the unit disk.

Three representations of the same operator, in increasing distance from the FEM
assembly:

``dtn_S_matrix``
    The Schur complement ``S = A_BB - A_BI A_II^-1 A_IB``, mapping nodal
    boundary voltages to the *weak* (Galerkin) flux ``f_B``. Real symmetric
    positive semi-definite, with the constants in its null space.
``dtn_map``
    The spectral map ``Lambda_hat = E* S E / (2 pi)`` taking Fourier
    coefficients of the voltage to Fourier coefficients of the current.
    Hermitian, and close to ``diag(|n|)`` when ``sigma == 1``.
``pointwise_flux``
    Nodal values of ``sigma du/dnu`` itself, obtained by inverting the boundary
    mass matrix against ``S g``.

The flux is never obtained by differentiating ``u`` at the boundary: the rows of
the stiffness matrix that ``dirichlet_system`` discards already hold it, to the
full accuracy of the discretization.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from stiffness_matrix import assemble_stiffness


def boundary_angles(mesh):
    """Polar angles of the boundary nodes, in ``mesh.boundary_indices`` order.

    ``triangulate_disk`` places the boundary ring at uniform angles
    ``2 pi b / B`` counter-clockwise from ``theta = 0``, so on the unit circle
    arclength and angle coincide. The angles are read back off the coordinates
    rather than regenerated, so the result stays correct if the mesh changes.

    Parameters
    ----------
    mesh : triangulation.DiskMesh
        Mesh providing ``points`` and ``boundary_indices``.

    Returns
    -------
    numpy.ndarray
        Length-``B`` array of angles in ``(-pi, pi]``.
    """
    B = np.asarray(mesh.boundary_indices, dtype=np.intp)
    return np.arctan2(mesh.points[1, B], mesh.points[0, B])


def synthesis_matrix(mesh, N):
    """The matrix ``E`` with ``E[b, n] = exp(i n theta_b)``, ``n = -N ... N``.

    ``E @ g_hat`` is the vector of boundary nodal values of
    ``g(theta) = sum_n g_hat_n exp(i n theta)`` -- the ``U_B`` that
    :func:`dtn_S_matrix` consumes -- and ``E.conj().T`` tests a boundary quantity
    against the interpolated modes ``exp(-i m theta)``.

    Parameters
    ----------
    mesh : triangulation.DiskMesh
        Mesh providing the boundary ring.
    N : int
        Highest Fourier mode retained.

    Returns
    -------
    numpy.ndarray
        Complex ``(B, 2*N + 1)`` array.

    Raises
    ------
    ValueError
        If ``2 * N >= B``. Discrete orthogonality of the boundary nodes needs
        ``|n - m| <= 2N < B``; in practice ``N <= B / 8`` keeps the P1
        representation of the modes accurate too.
    """
    N = _check_order(N)
    theta = boundary_angles(mesh)
    if 2 * N >= theta.size:
        raise ValueError(
            f"N={N} aliases on {theta.size} boundary nodes; need 2*N < B "
            f"(and N <~ B/8 for accuracy)"
        )
    return np.exp(1j * np.outer(theta, np.arange(-N, N + 1)))


def weak_flux(mesh, sigma, g):
    """Weak (Galerkin) Neumann data for one or more boundary voltages.

    Solves the interior Dirichlet problem for each column of ``g`` and returns
    the boundary rows of the residual,

    ``f_B = A_BB U_B - A_BI A_II^-1 A_IB U_B``,

    whose ``j``-th entry is ``int sigma du/dnu phi_j ds``. A single sparse LU
    factorization of ``A_II`` is reused across all columns.

    Parameters
    ----------
    mesh : triangulation.DiskMesh
        Mesh providing ``points``, ``triangles``, ``interior_nodes`` and
        ``boundary_indices``.
    sigma : float or array_like
        Conductivity, passed through to
        :func:`stiffness_matrix.assemble_stiffness` (scalar, or one value per
        triangle).
    g : array_like, shape (B,) or (B, m)
        Boundary nodal voltages, in ``mesh.boundary_indices`` order. May be
        complex.

    Returns
    -------
    numpy.ndarray
        Array of the same shape as ``g`` holding the weak flux.
    """
    g = np.asarray(g)
    if g.ndim not in (1, 2):
        raise ValueError("g must be a vector or a matrix of boundary data")

    A = assemble_stiffness(mesh, sigma)
    I = np.asarray(mesh.interior_nodes, dtype=np.intp)
    B = np.asarray(mesh.boundary_indices, dtype=np.intp)
    if g.shape[0] != B.size:
        raise ValueError(
            f"g has {g.shape[0]} rows, expected {B.size} boundary nodes"
        )

    G = g.reshape(g.shape[0], -1)
    # Blocks come out of CSR by a row slice followed by a column slice, the
    # same partition dirichlet_system uses.
    A_II = sp.csc_matrix(A[I][:, I])
    A_IB = A[I][:, B]
    A_BI = A[B][:, I]
    A_BB = A[B][:, B]

    lu = spla.splu(A_II)
    U_I = _lu_solve(lu, -(A_IB @ G))          # interior solution, one per column
    F = A_BB @ G + A_BI @ U_I
    return F.reshape(g.shape)


def dtn_S_matrix(mesh, sigma):
    """The discrete DtN map as the Schur complement ``S``.

    ``S = A_BB - A_BI A_II^-1 A_IB`` maps nodal boundary voltages to weak flux,
    ``f_B = S U_B``. Costs ``B`` solves against ``A_II``; when only a few
    Fourier modes are wanted, :func:`dtn_map` avoids forming it.

    Parameters
    ----------
    mesh : triangulation.DiskMesh
        Mesh providing the node partition.
    sigma : float or array_like
        Conductivity, as in :func:`weak_flux`.

    Returns
    -------
    numpy.ndarray
        Dense ``(B, B)`` real symmetric positive semi-definite array, with
        ``S @ ones(B) approx 0``.
    """
    B = np.asarray(mesh.boundary_indices, dtype=np.intp)
    return weak_flux(mesh, sigma, np.eye(B.size))


def dtn_map(mesh, sigma, N, S=None):
    """The DtN map in the Fourier basis: ``Lambda_hat = E* S E / (2 pi)``.

    With ``g(theta) = sum_n g_hat_n exp(i n theta)`` and the convention
    ``q_hat_m = (1 / 2 pi) int q(theta) exp(-i m theta) dtheta``, the current
    ``q = sigma du/dnu`` has coefficients ``q_hat = Lambda_hat @ g_hat``. The
    boundary mass matrix does not appear: ``f_B`` is already a set of pairings
    against boundary basis functions, and so is a Fourier coefficient.

    If ``S`` is not supplied the Schur complement is never formed -- the
    interior problem is solved once per mode instead, which is ``2*N + 1``
    solves against ``2*N + 1 << B``.

    Parameters
    ----------
    mesh : triangulation.DiskMesh
        Mesh providing the boundary ring and node partition.
    sigma : float or array_like
        Conductivity, as in :func:`weak_flux`. Ignored when ``S`` is given.
    N : int
        Highest Fourier mode retained; modes are ordered ``[-N, ..., 0, ..., N]``.
    S : array_like, shape (B, B), optional
        A precomputed :func:`dtn_S_matrix`, reused instead of re-solving.

    Returns
    -------
    numpy.ndarray
        Complex Hermitian ``(2*N + 1, 2*N + 1)`` array. For ``sigma == 1`` it is
        close to ``diag(|n|)``, and row/column ``n = 0`` vanish.
    """
    E = synthesis_matrix(mesh, N)
    F = np.asarray(S) @ E if S is not None else weak_flux(mesh, sigma, E)
    return (E.conj().T @ F) / (2 * np.pi)


def boundary_mass_matrix(mesh):
    """The P1 mass matrix on the boundary curve.

    ``M[i, j] = int_dOmega phi_i phi_j ds``, assembled over
    ``mesh.boundary_edges``; an edge of length ``L`` contributes the local
    matrix ``(L / 6) [[2, 1], [1, 2]]``. Indices are local to
    ``mesh.boundary_indices``, matching :func:`dtn_S_matrix`.

    Parameters
    ----------
    mesh : triangulation.DiskMesh
        Mesh providing ``points``, ``boundary_indices`` and ``boundary_edges``.

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetric positive definite ``(B, B)`` matrix. Its row sums are the
        nodal spacings, ``M @ ones(B) == h``.
    """
    points = mesh.points
    B = np.asarray(mesh.boundary_indices, dtype=np.intp)

    # boundary_edges holds global point indices; map them to positions in B.
    local = np.full(points.shape[1], -1, dtype=np.intp)
    local[B] = np.arange(B.size)
    a, b = np.asarray(mesh.boundary_edges, dtype=np.intp)
    ia, ib = local[a], local[b]
    if np.any(ia < 0) or np.any(ib < 0):
        raise ValueError("boundary_edges references a non-boundary node")

    L = np.hypot(points[0, b] - points[0, a], points[1, b] - points[1, a])
    rows = np.concatenate([ia, ia, ib, ib])
    cols = np.concatenate([ia, ib, ia, ib])
    vals = np.concatenate([2 * L, L, L, 2 * L]) / 6.0
    return sp.coo_matrix(
        (vals, (rows, cols)), shape=(B.size, B.size)
    ).tocsr()


def pointwise_flux(mesh, g, sigma=None, S=None):
    """Nodal values of the boundary current ``sigma du/dnu``.

    ``S g`` is the flux tested against each boundary basis function; solving
    ``M q = S g`` against the boundary mass matrix converts that to the
    coefficients of the piecewise-linear ``q_h`` carrying the same weak data,
    i.e. the L2 projection of the true flux onto the P1 trace space. Since the
    basis is nodal, ``q[j]`` is the flux at boundary node ``j`` -- this is the
    quantity to plot against arclength.

    Solving with ``M`` rather than dividing by lumped edge lengths is what keeps
    this second-order accurate.

    Parameters
    ----------
    mesh : triangulation.DiskMesh
        Mesh providing the boundary ring.
    g : array_like, shape (B,) or (B, m)
        Boundary nodal voltages.
    sigma : float or array_like, optional
        Conductivity. Required unless ``S`` is given.
    S : array_like, shape (B, B), optional
        A precomputed :func:`dtn_S_matrix`, reused instead of re-solving.

    Returns
    -------
    numpy.ndarray
        Nodal flux values, same shape as ``g``.
    """
    g = np.asarray(g)
    if S is not None:
        f_B = np.asarray(S) @ g
    elif sigma is not None:
        f_B = weak_flux(mesh, sigma, g)
    else:
        raise ValueError("supply either sigma or a precomputed S")

    M = sp.csc_matrix(boundary_mass_matrix(mesh))
    return _lu_solve(spla.splu(M), f_B)


def _check_order(N):
    """Validate a Fourier truncation order."""
    if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)):
        raise TypeError("N must be a nonnegative integer")
    N = int(N)
    if N < 0:
        raise ValueError("N must be nonnegative")
    return N


def _lu_solve(lu, W):
    """Apply a real sparse LU factorization to a possibly complex right side.

    ``SuperLU.solve`` wants real, C-contiguous, float64 columns, so a complex
    right-hand side is split and recombined. ``W.real`` is a strided view of a
    complex array, hence the copy.
    """
    if np.iscomplexobj(W):
        real = lu.solve(np.ascontiguousarray(W.real, dtype=np.float64))
        imag = lu.solve(np.ascontiguousarray(W.imag, dtype=np.float64))
        return real + 1j * imag
    return lu.solve(np.ascontiguousarray(W, dtype=np.float64))


if __name__ == "__main__":
    from triangulation import triangulate_disk

    mesh = triangulate_disk(20, 8)
    x, y = mesh.points[:, mesh.boundary_indices]
    B = x.size
    N = 8

    # --- The homogeneous case, where every answer is known in closed form. ---
    S = dtn_S_matrix(mesh, 1.0)
    print(f"{B} boundary nodes, {mesh.triangles.shape[1]} triangles\n")

    print("S = S^T                :", f"{np.abs(S - S.T).max():.3e}")
    print("S @ 1 = 0              :", f"{np.abs(S @ np.ones(B)).max():.3e}")

    # u = x is harmonic, so g = x has flux du/dnu = x and energy = area = pi.
    print("x^T S x = pi           :", f"{abs(x @ S @ x - np.pi):.3e}")
    q = pointwise_flux(mesh, x, S=S)
    print("pointwise flux of x = x:", f"{np.abs(q - x).max():.3e}")

    # M @ 1 = h, the nodal spacing, since the hat functions partition unity.
    M = boundary_mass_matrix(mesh)
    h = 2 * np.pi / B
    print("M @ 1 = h              :", f"{np.abs(M @ np.ones(B) - h).max():.3e}")

    # --- The spectral map, which should be diag(|n|). ---
    Lam = dtn_map(mesh, 1.0, N)
    exact = np.diag(np.abs(np.arange(-N, N + 1)).astype(float))
    print("\nLambda_hat Hermitian   :", f"{np.abs(Lam - Lam.conj().T).max():.3e}")
    print("off-diagonal error     :", f"{np.abs(Lam - np.diag(np.diag(Lam))).max():.3e}")
    print("diagonal error         :", f"{np.abs(np.diag(Lam) - np.diag(exact)).max():.3e}")

    # Forming S and solving mode-by-mode must agree exactly.
    print("with/without S agree   :", f"{np.abs(Lam - dtn_map(mesh, 1.0, N, S=S)).max():.3e}")

    print("\ndiag(Lambda_hat) vs |n|:")
    for n, got in zip(range(-N, N + 1), np.diag(Lam).real):
        print(f"  n = {n:+d}   {got:8.4f}   (exact {abs(n)})")

    # --- An inhomogeneous phantom: the map is no longer diagonal. ---
    from conductivity import gaussian_bumps, sample_sigma

    field = gaussian_bumps([((0.4, 0.0), 0.20, 1.5), ((-0.3, 0.3), 0.15, 1.0)])
    sigma = sample_sigma(mesh, field)
    Lam_sigma = dtn_map(mesh, sigma, N)
    print("\nphantom: Hermitian     :", f"{np.abs(Lam_sigma - Lam_sigma.conj().T).max():.3e}")
    print("phantom: |Lambda - Lambda_1| :", f"{np.abs(Lam_sigma - Lam).max():.4f}")
