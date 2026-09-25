
from dtn_map import dtn_map 
from faddeev_H_k import H_k_mat
import numpy as np 
def compute_psi(mesh, sigma, k, N):
    Lsigma = dtn_map(mesh, sigma, N)
    L1 = dtn_map(mesh, 1, N)
    S0 = np.array([1.0/n if n != 0 else 0  for n in range(-N,N+1)])
    Hk = H_k_mat(k,1,N)
    def a(n, k):
        if n < 0:
            return 0
        p = 1
        for m in range(n-1):
            p *= (1j*k)/(m+1) 

        return p 
    Eikz = np.array([a(n,k) for n in range(-N,N+1)])
    Id = np.eye(2*N+1)
    
    breakpoint()
    A = Id + (S0 + Hk)@(Lsigma - L1)
    psi = np.linalg.solve(A, Eikz)
    return psi 


from triangulation import triangulate_disk
from conductivity import gaussian_bumps, sample_sigma
disk_mesh = triangulate_disk(5,5)
field = gaussian_bumps(
    [
        ((0.4, 0.0), 0.20, 1.5),    # high-conductivity blob, right
        ((-0.3, 0.3), 0.15, 1.0),   # smaller blob, upper-left
        ((0.0, -0.4), 0.25, 0.6),  # low-conductivity dip, bottom
    ]
)

sigma = sample_sigma(disk_mesh, field)
psi = compute_psi(disk_mesh, sigma, 1, 5)
print(psi)