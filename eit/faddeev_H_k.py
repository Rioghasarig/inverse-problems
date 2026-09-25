
"""The regular Faddeev Green-function boundary operator on a circle."""

from math import comb, pi

import numpy as np

def H_k_mat(k, R, N):
    """Assemble the truncated regular Faddeev single-layer operator as a matrix.

    Returns the matrix ``M`` with ``M @ phi_f == H_k(phi_f, k, R, N)``.  Rows
    and columns are both in centered Fourier order ``[-N, ..., 0, ..., N]``:
    entry ``M[N + j, N + m]`` is the coefficient of ``phi_hat(m)`` in ``c_j``.

    Parameters
    ----------
    k : complex
        Nonzero Faddeev parameter.
    R : float
        Positive radius of the boundary circle.
    N : int
        Taylor/Fourier truncation order.

    Returns
    -------
    ndarray, shape (2*N + 1, 2*N + 1)
    """
    if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)):
        raise TypeError("N must be a nonnegative integer")
    N = int(N)
    if N < 0:
        raise ValueError("N must be nonnegative")
    if k == 0:
        raise ValueError("H_k is undefined for k=0")
    if not np.isfinite(k):
        raise ValueError("k must be finite")
    if not np.isfinite(R) or R <= 0:
        raise ValueError("R must be a positive finite number")

    dtype = np.result_type(k, np.complex128)
    coefficients = np.zeros((2 * N + 1, 2 * N + 1), dtype=dtype)

    # H_k(z) = Re(sum(alpha[n] z**n)).  The recurrence avoids explicit
    # complex powers and factorials.
    alpha = np.empty(N + 1, dtype=dtype)
    euler_gamma = 0.5772156649015328606
    alpha[0] = -(euler_gamma + np.log(abs(k))) / (2 * pi)
    if N:
        alpha[1] = -(1j * k) / (2 * pi)
        for n in range(2, N + 1):
            alpha[n] = alpha[n - 1] * (1j * k) * (n - 1) / (n * n)

    # The n = 0 terms both land in column N, so these must accumulate.
    for n in range(N + 1):
        factor = pi * R ** (n + 1) * (-1) ** n
        coefficients[N, N - n] += factor * alpha[n]
        coefficients[N, N + n] += factor * np.conjugate(alpha[n])

    for j in range(1, N + 1):
        for n in range(j, N + 1):
            factor = pi * R ** (n + 1) * comb(n, j) * (-1) ** (n - j)
            coefficients[N + j, N + j - n] = factor * alpha[n]
            coefficients[N - j, N + n - j] = factor * np.conjugate(alpha[n])

    return coefficients


def H_k(phi_f, k, R, N=None):
    """Apply the truncated regular Faddeev single-layer operator.

    ``phi_f`` contains the Fourier coefficients in centered order
    ``[-N, ..., 0, ..., N]``.  The returned array uses the same ordering and
    contains the coefficients of ``(H_k * phi)(theta)``.  If ``N`` is omitted,
    it is inferred from the (necessarily odd) length of ``phi_f``.

    Parameters
    ----------
    phi_f : array_like, shape (2*N + 1,)
        Centered Fourier coefficients of the boundary density.
    k : complex
        Nonzero Faddeev parameter.
    R : float
        Positive radius of the boundary circle.
    N : int, optional
        Taylor/Fourier truncation order.
    """
    phi_f = np.asarray(phi_f)
    if phi_f.ndim != 1:
        raise ValueError("phi_f must be a one-dimensional coefficient array")

    if N is None:
        if phi_f.size % 2 != 1:
            raise ValueError("phi_f must have odd length when N is omitted")
        N = (phi_f.size - 1) // 2
    if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)):
        raise TypeError("N must be a nonnegative integer")
    N = int(N)
    if N < 0:
        raise ValueError("N must be nonnegative")
    if phi_f.size != 2 * N + 1:
        raise ValueError("phi_f must contain exactly 2*N + 1 coefficients")
    if k == 0:
        raise ValueError("H_k is undefined for k=0")
    if not np.isfinite(k):
        raise ValueError("k must be finite")
    if not np.isfinite(R) or R <= 0:
        raise ValueError("R must be a positive finite number")

    dtype = np.result_type(phi_f.dtype, k, np.complex128)
    phi_f = phi_f.astype(dtype, copy=False)
    coefficients = np.zeros(2 * N + 1, dtype=dtype)

    # H_k(z) = Re(sum(alpha[n] z**n)).  The recurrence avoids explicit
    # complex powers and factorials.
    alpha = np.empty(N + 1, dtype=dtype)
    euler_gamma = 0.5772156649015328606
    alpha[0] = -(euler_gamma + np.log(abs(k))) / (2 * pi)
    if N:
        alpha[1] = -(1j * k) / (2 * pi)
        for n in range(2, N + 1):
            alpha[n] = alpha[n - 1] * (1j * k) * (n - 1) / (n * n)

    for n in range(N + 1):
        factor = pi * R ** (n + 1) * (-1) ** n
        coefficients[N] += factor * (
            alpha[n] * phi_f[N - n]
            + np.conjugate(alpha[n]) * phi_f[N + n]
        )

    for j in range(1, N + 1):
        c_pos = dtype.type(0)
        c_neg = dtype.type(0)
        for n in range(j, N + 1):
            factor = pi * R ** (n + 1) * comb(n, j) * (-1) ** (n - j)
            c_pos += factor * alpha[n] * phi_f[N + j - n]
            c_neg += factor * np.conjugate(alpha[n]) * phi_f[N + n - j]
        coefficients[N + j] = c_pos
        coefficients[N - j] = c_neg

    return coefficients
