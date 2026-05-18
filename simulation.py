#!/usr/bin/env python3
# opinion_hierarchy_three_species_mpi.py
import os, sys
import numpy as np
from mpi4py import MPI
from numba import njit
import time 
# --------------------- user params (submissive managers) ---------------------
N1, N2, N3 = 16000, 4000, 200          # workers, managers, CEOs
T          = 0.5
DT         = 1e-3                 # keep reasonable for MPI run-time
SIGMA      = 0.0                  # additive noise amplitude
SAVE_EVERY = 1
OUTDIR     = "data"

R1, R2, R3 = 1.0, 2.5, 5.0         # interaction radii (depend on source species j)
D11, D12, D22, D23, D33 = 5.0, 10.0, 2.0,  5.0, 1   # influence matrix entries

# --------------------- initial conditions ---------------------
def init_opinion(N, seed=0):
    rng = np.random.default_rng(seed)
    num_gaussian = rng.integers(low=2, high=9, size=1)[0]
    value = rng.uniform(low=-5, high=5, size=num_gaussian)
    p = rng.dirichlet(np.ones(num_gaussian), size=1)[0]
    x = 2*rng.standard_normal(N) + rng.choice(value, size=N, p=p)
    x = x - np.mean(x, axis=0)
    return x.astype(np.float64)

# --------------------- kernel φ and K_{ij} ---------------------
def phi_bump(u):
    a = np.abs(u)
    out = np.zeros_like(a)
    mask = a < 1.0
    if np.any(mask):
        um = a[mask]
        out[mask] = np.exp(1.0 - 1.0 / (1.0 - um**10))
    return out

def K_scalar(z, Dij, Rj):
    # 1D: K_ij(z) = -D_ij * φ(z/R_j) * z
    u = z / Rj
    return -Dij * phi_bump(u) * z

# --------------------- neighbor search (1D window via sorting) ---------------------
# def drift_ij(x_i, x_j, Dij, Rj, exclude_self=False):
#     """
#     Contribution to dx_i/dt from species j:
#       (1/den) * sum_{|x_i - x_j| < Rj} K_ij(x_i - x_j),
#     with den = N_j  (or N_j - 1 for intra-species if exclude_self=True).
#     x_i can be a local slice; x_j is the full array of species j.
#     """
#     Ni, Nj = x_i.size, x_j.size
#     if Nj == 0:
#         return np.zeros_like(x_i)

#     # order = np.argsort(x_j, kind='mergesort')
#     # xs = x_j[order]

#     den = Nj - 1 if exclude_self and Nj > 1 else Nj
#     den = max(den, 1)

#     out = np.zeros_like(x_i)
#     for idx in range(Ni):
#         xi = x_i[idx]
#         z    = xi - x_j
#         out[idx] = K_scalar(z, Dij, Rj).sum() / den

#     # z = x_i[:, None] - x_j[None, :]  # (Ni, Nj)
#     # out = K_scalar(z, Dij, Rj).sum(axis=1) / den
#     return out
@njit
def drift_ij_numba(x_i, x_j, Dij, Rj, exclude_self):
    Ni = x_i.shape[0]
    Nj = x_j.shape[0]
    den = (Nj - 1) if (exclude_self and Nj > 1) else Nj
    if den < 1:
        den = 1
    out = np.zeros(Ni)
    for i in range(Ni):
        s = 0.0
        for j in range(Nj):
            z = x_i[i] - x_j[j]
            u = abs(z) / Rj
            if u < 1.0:
                k = -Dij * np.exp(1.0 - 1.0 / (1.0 - u**10)) * z
                s += k
        out[i] = s / den
    return out
# --------------------- MPI helpers ---------------------
def split_slice(N, size, rank):
    counts = np.full(size, N // size, dtype=int)
    counts[: N % size] += 1
    offsets = np.zeros(size, dtype=int); offsets[1:] = np.cumsum(counts)[:-1]
    i0 = offsets[rank]; i1 = i0 + counts[rank]
    return slice(i0, i1), counts, offsets

def allgatherv_1d(local, counts, offsets, comm):
    Ntot = int(np.sum(counts))
    full = np.empty(Ntot, dtype=np.float64)
    comm.Allgatherv(
        [local, MPI.DOUBLE],
        [full, counts.astype(np.int32), offsets.astype(np.int32), MPI.DOUBLE]
    )
    return full

# --------------------- main ---------------------
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(sys.argv) < 2:
        if rank == 0:
            print("Usage: mpirun -np P python opinion_hierarchy_three_species_mpi.py <seed:int>")
        sys.exit(1)
    seed = int(sys.argv[1])

    # init RNG per rank (independent noise streams)
    rng = np.random.default_rng(seed + 1000*rank)

    steps = int(np.round(T / DT))
    nsave = 1 + steps // SAVE_EVERY

    # initial conditions (rank 0), then Bcast to everyone
    if rank == 0:
        x1 = init_opinion(N1, seed=seed)
        x2 = init_opinion(N2, seed=seed+1)
        x3 = init_opinion(N3, seed=seed+2)
    else:
        x1 = np.empty(N1, dtype=np.float64)
        x2 = np.empty(N2, dtype=np.float64)
        x3 = np.empty(N3, dtype=np.float64)

    comm.Bcast(x1, root=0)
    comm.Bcast(x2, root=0)
    comm.Bcast(x3, root=0)

    # each species gets its own slice per rank
    sl1, cnt1, off1 = split_slice(N1, size, rank)
    sl2, cnt2, off2 = split_slice(N2, size, rank)
    sl3, cnt3, off3 = split_slice(N3, size, rank)

    # rank 0 save buffers
    if rank == 0:
        X1_save = np.empty((nsave, N1), dtype=np.float64)
        X2_save = np.empty((nsave, N2), dtype=np.float64)
        X3_save = np.empty((nsave, N3), dtype=np.float64)
        X1_save[0], X2_save[0], X3_save[0] = x1.copy(), x2.copy(), x3.copy()
        frame = 1
    else:
        X1_save = X2_save = X3_save = None
        frame = None

    # time loop
    sqrt_dt_sigma = SIGMA * np.sqrt(DT)
    for t in range(1, steps + 1):
        if rank == 0:
            print(f"[rank {rank}] step {t}/{steps}", flush=True)

        dx1_loc = drift_ij_numba(x1[sl1], x1, D11, R1, exclude_self=True) \
                + drift_ij_numba(x1[sl1], x2, D12, R2, exclude_self=False)
        dx2_loc = drift_ij_numba(x2[sl2], x3, D23, R3, exclude_self=False)\
                + drift_ij_numba(x2[sl2], x2, D22, R2, exclude_self=True)
        dx3_loc = drift_ij_numba(x3[sl3], x3, D33, R3, exclude_self=True)



        # Euler–Maruyama updates (local slices only)
        if SIGMA > 0.0:
            x1_loc = x1[sl1] + DT * dx1_loc + sqrt_dt_sigma * rng.standard_normal(x1[sl1].shape)
            x2_loc = x2[sl2] + DT * dx2_loc + sqrt_dt_sigma * rng.standard_normal(x2[sl2].shape)
            x3_loc = x3[sl3] + DT * dx3_loc + sqrt_dt_sigma * rng.standard_normal(x3[sl3].shape)
        else:
            x1_loc = x1[sl1] + DT * dx1_loc
            x2_loc = x2[sl2] + DT * dx2_loc
            x3_loc = x3[sl3] + DT * dx3_loc

        # allgather local → full for next step (and optional saving)
        x1 = allgatherv_1d(x1_loc, cnt1, off1, comm)
        x2 = allgatherv_1d(x2_loc, cnt2, off2, comm)
        x3 = allgatherv_1d(x3_loc, cnt3, off3, comm)

        if (t % SAVE_EVERY == 0) and (rank == 0):
            X1_save[frame], X2_save[frame], X3_save[frame] = x1.copy(), x2.copy(), x3.copy()
            frame += 1

    # save on rank 0
    if rank == 0:
        os.makedirs(OUTDIR, exist_ok=True)
        np.save(os.path.join(OUTDIR, f"opinion_workers_mpi_{seed}.npy"),  X1_save)
        np.save(os.path.join(OUTDIR, f"opinion_managers_mpi_{seed}.npy"), X2_save)
        np.save(os.path.join(OUTDIR, f"opinion_ceos_mpi_{seed}.npy"),     X3_save)
        print("[OK] saved:",
              f"{OUTDIR}/opinion_workers_mpi_{seed}.npy  {X1_save.shape}",
              f"{OUTDIR}/opinion_managers_mpi_{seed}.npy {X2_save.shape}",
              f"{OUTDIR}/opinion_ceos_mpi_{seed}.npy     {X3_save.shape}",
              sep="\n")

if __name__ == "__main__":
    main()
