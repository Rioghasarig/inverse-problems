import numpy as np 


def fourier_coeffs(g,N):
    k = np.arange(N)
    ws = np.exp(-2j * np.pi * k[:, None] * k[None, :] / N)
    a = ws @ g / N
    return a

def fourier_coeffs_2d(g,N):
    k = np.arange(N)
    ws1 = np.exp(-2j * np.pi * k[:, None] * k[None, :] / N)
    ws2 = np.kron(ws1,ws1)
    gvec = g.flatten()
    return (ws2 @ gvec).reshape(N,N) / N**2


def fourier_eval(a,N):
    k = np.arange(N)
    ws = np.exp(2j * np.pi * k[:, None] * k[None, :] / N)
    g = ws @ a
    return g 

def fourier_eval_2d(a,N):
    k = np.arange(N) 
    ws1 = np.exp(2j * np.pi * k[:, None] * k[None, :] / N)
    ws2 = np.kron(ws1,ws1)
    avec = a.flatten()
    return (ws2@avec).reshape(N,N)

def convolution(g,h):
    k = np.arange(len(g))
    gs = g[(k[:, None] - k[None, :]) % len(g)]
    return gs @ h
def convolution_2d(g,h,N):
    k = np.arange(N)
    gs = g[(k[:,None, None, None] - k[None,:,None,None])%N, (k[None,None,:,None] - k[None,None,None,:])%N]
    breakpoint()
    return np.einsum('ikjl,kl',gs, h)

def convolve_fft(f_vec,g_vec,s, N):
    h_vec = 2*s/N*fourier_eval(N*fourier_coeffs(f_vec,N)*fourier_coeffs(g_vec,N),N)
    return h_vec 


def apply_A(v, q, s):
    """Apply I + convolution to samples on [-s, s), with even length."""
    v = np.asarray(v)
    q = np.asarray(q)
    if v.ndim != 1 or q.shape != v.shape or len(v) == 0:
        raise ValueError("v and q must be nonempty vectors of equal length")
    if s <= 0 or len(v) % 2:
        raise ValueError("s must be positive and the sample count must be even")
    c = convolve_fft(q, v, s, len(v))
    return v + np.roll(c, -(len(v) // 2))


def solve_gmres(q, s, b=None, x0=None, rtol=1e-10, atol=0.0,
                restart=20, maxiter=100):
    """Solve m + circular_convolution(g, m) = b using SciPy GMRES.

    q contains kernel samples on [-s, s), with an even sample count.
    b defaults to ones and x0 defaults to zero. maxiter counts restart
    cycles; restart is the maximum number of inner iterations per cycle.

    Returns (m, diagnostics). diagnostics includes SciPy's info (zero means
    convergence), the independently computed residual norm, and the inner
    relative residual history. Always check diagnostics['converged'].
    """
    from scipy.sparse.linalg import LinearOperator, gmres

    q = np.asarray(q, dtype=complex)
    if q.ndim != 1 or q.size == 0 or q.size % 2:
        raise ValueError("q must be a nonempty vector of even length")
    if not np.isfinite(s) or s <= 0:
        raise ValueError("s must be finite and positive")
    b = np.ones(q.size, dtype=complex) if b is None else np.asarray(b, dtype=complex)
    if b.shape != q.shape:
        raise ValueError("b must have the same shape as q")
    if x0 is not None:
        x0 = np.asarray(x0, dtype=complex)
        if x0.shape != q.shape:
            raise ValueError("x0 must have the same shape as q")
    if not all(np.all(np.isfinite(v)) for v in (q, b) + (() if x0 is None else (x0,))):
        raise ValueError("q, b, and x0 must contain finite values")

    operator = LinearOperator((q.size, q.size),
                              matvec=lambda v: apply_A(v, q, s),
                              dtype=np.complex128)
    history = []
    m, info = gmres(operator, b, x0=x0, rtol=rtol, atol=atol,
                    restart=restart, maxiter=maxiter,
                    callback=history.append, callback_type="pr_norm")
    residual_norm = float(np.linalg.norm(b - apply_A(m, q, s)))
    b_norm = float(np.linalg.norm(b))
    tolerance = max(atol, rtol * b_norm)
    return m, {
        "info": int(info),
        "converged": bool(info == 0 and residual_norm <= tolerance),
        "iterations": len(history),
        "residual_norm": residual_norm,
        "relative_residual": residual_norm / b_norm if b_norm else None,
        "residual_history": np.asarray(history),
    }


def apply_A_direct(x, m, kernel, s):
    """Evaluate m(x) + integral g(wrap(x-y))*m(y) dy by adaptive quadrature.

    m and kernel are scalar callables; real and complex values are supported.
    This uses the continuous functions, independently of the sampled FFT sum.
    """
    from scipy.integrate import quad

    if s <= 0:
        raise ValueError("s must be positive")
    x = np.asarray(x, dtype=float)
    result = np.empty(x.shape, dtype=complex)
    for index in np.ndindex(x.shape):
        xi = (x[index] + s) % (2 * s) - s

        def integrand(y):
            lag = (xi - y + s) % (2 * s) - s
            return kernel(lag) * m(y)

        # Split at the point where the wrapped kernel argument jumps.
        points = [p for p in (xi - s, xi + s) if -s < p < s]
        real = quad(lambda y: np.real(integrand(y)), -s, s,
                    points=points, epsabs=1e-10, epsrel=1e-10)[0]
        imag = quad(lambda y: np.imag(integrand(y)), -s, s,
                    points=points, epsabs=1e-10, epsrel=1e-10)[0]
        result[index] = m(xi) + real + 1j * imag
    return np.real_if_close(result)


def f(x):
    return x**2 

def g(x):
    return np.exp(-x**2/2)

def plot_comparison(s=3.0, N=128, output="apply_A_comparison.png", show=True):
    """Compare both operators using m(x)=x**2 and g(x)=exp(-x**2/2)."""
    import matplotlib.pyplot as plt

    grid = -s + np.arange(N) * (2 * s / N)
    fft_values = apply_A(f(grid), g(grid), s)
    direct_values = apply_A_direct(grid, f, g, s)
    error = np.max(np.abs(fft_values - direct_values))
    print(f"N={N}, s={s:g}: max absolute difference = {error:.6e}")
    print(f"Maximum FFT imaginary part = {np.max(np.abs(fft_values.imag)):.3e}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for ax, values, title in zip(
        axes, (fft_values.real, direct_values),
        ("apply_A: convolve_fft + shift", "Direct adaptive quadrature"),
    ):
        ax.plot(grid, values, label="L(m)")
        ax.plot(grid, f(grid), "--", alpha=0.6, label="m(x) = x²")
        ax.set(xlabel="x", title=title)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Function value")
    fig.suptitle(f"g(x) = exp(-x²/2), s = {s:g}, N = {N}; max difference = {error:.3g}")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return grid, fft_values, direct_values


if __name__ == "__main__":
    s = 3.0
    N = 128
    grid = -s + np.arange(N) * (2 * s / N)
    m, diagnostics = solve_gmres(g(grid), s)
    print("GMRES:", diagnostics)
    if not diagnostics["converged"]:
        raise RuntimeError("GMRES did not reach the requested residual tolerance")
    exact_constant = 1 / (1 + (2 * s / N) * np.sum(g(grid)))
    print(f"Max error against discrete constant solution: {np.max(np.abs(m - exact_constant)):.3e}")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grid, m.real, label="GMRES solution")
    if np.max(np.abs(m.imag)) > 1e-10:
        ax.plot(grid, m.imag, "--", label="Imaginary part")
    ax.set(xlabel="x", ylabel="m(x)",
           title="GMRES solution of m + circular convolution(g, m) = 1")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("gmres_solution.png", dpi=160)
    plt.show()
    plt.close(fig)
