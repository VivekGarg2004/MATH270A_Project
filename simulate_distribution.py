#!/usr/bin/env python3
# simulate_distribution.py — Dirichlet/multinomial species counts + inlined MPI opinion dynamics
#
# Run with MPI from project root, e.g.:
#   mpirun -np 4 python simulate_distribution.py
#
# Outputs (relative to distribution root, default data_distribution/):
#   data/       — opinion_workers/managers/ceos_mpi_<seed>.npy (trajectories, sampled cohort sizes)
#   metadata/   — distribution_meta_mpi_<seed>.json, sampled_species_counts_mpi_<seed>.npy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from mpi4py import MPI
from numba import njit

# --------------------- layout under distribution root -----------------------------------------
DATA_DISTRIBUTION_DIR = "data_distribution"
DATA_SUBDIR = "data"
METADATA_SUBDIR = "metadata"


def distribution_data_dir(root: Union[str, Path]) -> Path:
    return Path(root) / DATA_SUBDIR


def distribution_metadata_dir(root: Union[str, Path]) -> Path:
    return Path(root) / METADATA_SUBDIR


# --------------------- opinion dynamics params (same as simulation.py) ------------------------
T = 0.5
DT = 1e-3
SIGMA = 0.0
SAVE_EVERY = 1
R1, R2, R3 = 1.0, 2.5, 5.0
D11, D12, D22, D23, D33 = 5.0, 10.0, 2.0, 5.0, 1.0

# --------------------- distribution defaults --------------------------------------------------
DEFAULT_TARGET_COUNTS = np.array([16000, 4000, 200], dtype=np.int64)
P_EVEN = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)

SWEEP_T_NUM = 11
SWEEP_CONCENTRATIONS = (10.0, 50.0, 200.0)

ArrayLike = Union[Sequence[float], np.ndarray]


# --------------------- interpolation ----------------------------------------------------------
def compute_interpolated_distribution(
    t: float,
    target_counts: ArrayLike,
    total_samples: int,
) -> np.ndarray:
    t = float(np.clip(t, 0.0, 1.0))
    target_counts = np.asarray(target_counts, dtype=np.int64).ravel()
    if target_counts.size != 3:
        raise ValueError("target_counts must have length 3 (species A, B, C).")
    s = int(target_counts.sum())
    if s != int(total_samples):
        raise ValueError(
            f"sum(target_counts)={s} must equal total_samples={int(total_samples)}."
        )
    p_target = target_counts.astype(np.float64) / float(total_samples)
    p_interp = (1.0 - t) * P_EVEN + t * p_target
    p_interp = np.maximum(p_interp, 0.0)
    p_interp /= p_interp.sum()
    return p_interp


# --------------------- Dirichlet layer --------------------------------------------------------
def _dirichlet_probabilities(
    t: float,
    target_counts: ArrayLike,
    total_samples: int,
    concentration: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if concentration <= 0:
        raise ValueError("concentration must be > 0.")
    p_interp = compute_interpolated_distribution(t, target_counts, total_samples)
    alpha = float(concentration) * p_interp
    alpha = np.maximum(alpha, 1e-12)
    p_sampled = rng.dirichlet(alpha)
    return p_interp, p_sampled


def sample_species_distribution(
    t: float,
    target_counts: ArrayLike,
    total_samples: int,
    concentration: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng if rng is not None else np.random.default_rng()
    _, p_sampled = _dirichlet_probabilities(
        t, target_counts, total_samples, concentration, rng
    )
    return p_sampled


# --------------------- multinomial + floors ---------------------------------------------------
def _reconcile_minimum_counts(
    counts: np.ndarray, total: int, floor: int
) -> np.ndarray:
    counts = np.maximum(counts.astype(np.int64), floor)
    diff = total - int(counts.sum())
    if diff == 0:
        return counts
    if diff > 0:
        counts[int(np.argmax(counts))] += diff
        return counts
    d = -diff
    while d > 0:
        idx = int(np.argmax(counts - floor))
        if counts[idx] <= floor:
            break
        take = min(d, counts[idx] - floor)
        counts[idx] -= take
        d -= take
    if d > 0:
        raise ValueError(
            "Could not reconcile counts with given floor; reduce floor or total."
        )
    return counts


def sample_species_counts(
    t: float,
    target_counts: ArrayLike,
    total_samples: int,
    concentration: float,
    rng: Optional[np.random.Generator] = None,
    *,
    min_per_species: int = 1,
) -> np.ndarray:
    rng = rng if rng is not None else np.random.default_rng()
    _, p_sampled = _dirichlet_probabilities(
        t, target_counts, total_samples, concentration, rng
    )
    counts = rng.multinomial(int(total_samples), p_sampled)
    counts = _reconcile_minimum_counts(counts, int(total_samples), min_per_species)
    return counts.astype(np.int64)


# --------------------- diagnostics & sweeps -------------------------------------------------
@dataclass
class SpeciesAllocationDiagnostics:
    t: float
    concentration: float
    p_interp: np.ndarray
    p_sampled: np.ndarray
    counts: np.ndarray
    total_samples: int

    def __str__(self) -> str:
        lines = [
            f"t={self.t:.4f}  concentration={self.concentration:.4g}",
            f"Interpolated distribution: {np.array2string(self.p_interp, precision=4)}",
            f"Dirichlet sampled distribution: {np.array2string(self.p_sampled, precision=4)}",
            f"Final counts: {np.array2string(self.counts.astype(int), separator=', ')} "
            f"(sum={int(self.counts.sum())})",
        ]
        return "\n".join(lines)


def sample_species_counts_with_diagnostics(
    t: float,
    target_counts: ArrayLike,
    total_samples: int,
    concentration: float,
    rng: Optional[np.random.Generator] = None,
    *,
    min_per_species: int = 1,
    verbose: bool = False,
) -> Tuple[np.ndarray, SpeciesAllocationDiagnostics]:
    rng = rng if rng is not None else np.random.default_rng()
    p_interp, p_sampled = _dirichlet_probabilities(
        t, target_counts, total_samples, concentration, rng
    )
    counts = rng.multinomial(int(total_samples), p_sampled)
    counts = _reconcile_minimum_counts(counts, int(total_samples), min_per_species)
    diag = SpeciesAllocationDiagnostics(
        t=t,
        concentration=concentration,
        p_interp=p_interp,
        p_sampled=p_sampled,
        counts=counts.astype(np.int64),
        total_samples=int(total_samples),
    )
    if verbose:
        print(diag, flush=True)
    return counts.astype(np.int64), diag


def sweep_t_concentration(
    t_values: Iterable[float],
    concentrations: Iterable[float],
    target_counts: Optional[ArrayLike] = None,
    *,
    total_samples: Optional[int] = None,
    counts_seed: int = 0,
    realizations: int = 1,
) -> Iterator[Tuple[float, float, int, np.ndarray, SpeciesAllocationDiagnostics]]:
    target = (
        np.asarray(target_counts, dtype=np.int64).ravel()
        if target_counts is not None
        else DEFAULT_TARGET_COUNTS.copy()
    )
    total = int(total_samples) if total_samples is not None else int(target.sum())
    if int(target.sum()) != total:
        raise ValueError("target_counts must sum to total_samples.")

    for ti, t in enumerate(t_values):
        for ci, concentration in enumerate(concentrations):
            for r in range(int(realizations)):
                sub_seed = int(counts_seed) + 100_003 * ti + 10_009 * ci + r
                rng = np.random.default_rng(sub_seed)
                counts, diag = sample_species_counts_with_diagnostics(
                    t,
                    target,
                    total,
                    float(concentration),
                    rng=rng,
                    verbose=False,
                )
                yield float(t), float(concentration), r, counts, diag


# --------------------- opinion dynamics (inlined from simulation.py) -------------------------
def init_opinion(N: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    num_gaussian = rng.integers(low=2, high=9, size=1)[0]
    value = rng.uniform(low=-5, high=5, size=num_gaussian)
    p = rng.dirichlet(np.ones(num_gaussian), size=1)[0]
    x = 2 * rng.standard_normal(N) + rng.choice(value, size=N, p=p)
    x = x - np.mean(x, axis=0)
    return x.astype(np.float64)


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


def split_slice(N: int, size: int, rank: int):
    counts = np.full(size, N // size, dtype=int)
    counts[: N % size] += 1
    offsets = np.zeros(size, dtype=int)
    offsets[1:] = np.cumsum(counts)[:-1]
    i0 = offsets[rank]
    i1 = i0 + counts[rank]
    return slice(i0, i1), counts, offsets


def allgatherv_1d(local, counts, offsets, comm):
    Ntot = int(np.sum(counts))
    full = np.empty(Ntot, dtype=np.float64)
    comm.Allgatherv(
        [local, MPI.DOUBLE],
        [full, counts.astype(np.int32), offsets.astype(np.int32), MPI.DOUBLE],
    )
    return full


def run_opinion_mpi_integration(
    seed: int,
    n1: int,
    n2: int,
    n3: int,
    trajectory_outdir: Path,
    *,
    verbose: bool = True,
) -> None:
    """
    MPI-parallel opinion integration; rank 0 writes three .npy files under trajectory_outdir.
    Must be called collectively on all ranks in MPI.COMM_WORLD.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N1, N2, N3 = int(n1), int(n2), int(n3)
    rng = np.random.default_rng(seed + 1000 * rank)

    steps = int(np.round(T / DT))
    nsave = 1 + steps // SAVE_EVERY

    if rank == 0:
        x1 = init_opinion(N1, seed=seed)
        x2 = init_opinion(N2, seed=seed + 1)
        x3 = init_opinion(N3, seed=seed + 2)
    else:
        x1 = np.empty(N1, dtype=np.float64)
        x2 = np.empty(N2, dtype=np.float64)
        x3 = np.empty(N3, dtype=np.float64)

    comm.Bcast(x1, root=0)
    comm.Bcast(x2, root=0)
    comm.Bcast(x3, root=0)

    sl1, cnt1, off1 = split_slice(N1, size, rank)
    sl2, cnt2, off2 = split_slice(N2, size, rank)
    sl3, cnt3, off3 = split_slice(N3, size, rank)

    if rank == 0:
        X1_save = np.empty((nsave, N1), dtype=np.float64)
        X2_save = np.empty((nsave, N2), dtype=np.float64)
        X3_save = np.empty((nsave, N3), dtype=np.float64)
        X1_save[0], X2_save[0], X3_save[0] = x1.copy(), x2.copy(), x3.copy()
        frame = 1
    else:
        X1_save = X2_save = X3_save = None
        frame = None

    sqrt_dt_sigma = SIGMA * np.sqrt(DT)
    for t in range(1, steps + 1):
        if rank == 0 and verbose:
            print(f"[rank {rank}] step {t}/{steps}", flush=True)

        dx1_loc = drift_ij_numba(x1[sl1], x1, D11, R1, True) + drift_ij_numba(
            x1[sl1], x2, D12, R2, False
        )
        dx2_loc = drift_ij_numba(x2[sl2], x3, D23, R3, False) + drift_ij_numba(
            x2[sl2], x2, D22, R2, True
        )
        dx3_loc = drift_ij_numba(x3[sl3], x3, D33, R3, True)

        if SIGMA > 0.0:
            x1_loc = x1[sl1] + DT * dx1_loc + sqrt_dt_sigma * rng.standard_normal(
                x1[sl1].shape
            )
            x2_loc = x2[sl2] + DT * dx2_loc + sqrt_dt_sigma * rng.standard_normal(
                x2[sl2].shape
            )
            x3_loc = x3[sl3] + DT * dx3_loc + sqrt_dt_sigma * rng.standard_normal(
                x3[sl3].shape
            )
        else:
            x1_loc = x1[sl1] + DT * dx1_loc
            x2_loc = x2[sl2] + DT * dx2_loc
            x3_loc = x3[sl3] + DT * dx3_loc

        x1 = allgatherv_1d(x1_loc, cnt1, off1, comm)
        x2 = allgatherv_1d(x2_loc, cnt2, off2, comm)
        x3 = allgatherv_1d(x3_loc, cnt3, off3, comm)

        if (t % SAVE_EVERY == 0) and (rank == 0):
            X1_save[frame], X2_save[frame], X3_save[frame] = (
                x1.copy(),
                x2.copy(),
                x3.copy(),
            )
            frame += 1

    if rank == 0:
        trajectory_outdir.mkdir(parents=True, exist_ok=True)
        np.save(trajectory_outdir / f"opinion_workers_mpi_{seed}.npy", X1_save)
        np.save(trajectory_outdir / f"opinion_managers_mpi_{seed}.npy", X2_save)
        np.save(trajectory_outdir / f"opinion_ceos_mpi_{seed}.npy", X3_save)
        if verbose:
            print(
                "[OK] saved:",
                f"{trajectory_outdir}/opinion_workers_mpi_{seed}.npy  {X1_save.shape}",
                f"{trajectory_outdir}/opinion_managers_mpi_{seed}.npy {X2_save.shape}",
                f"{trajectory_outdir}/opinion_ceos_mpi_{seed}.npy     {X3_save.shape}",
                sep="\n",
                flush=True,
            )

    comm.Barrier()


# --------------------- metadata ---------------------------------------------------------------
def _trajectory_paths_relative(seed: int) -> List[str]:
    return [
        f"{DATA_SUBDIR}/opinion_workers_mpi_{seed}.npy",
        f"{DATA_SUBDIR}/opinion_managers_mpi_{seed}.npy",
        f"{DATA_SUBDIR}/opinion_ceos_mpi_{seed}.npy",
    ]


def build_distribution_metadata(
    seed: int,
    diag: SpeciesAllocationDiagnostics,
    target_counts: np.ndarray,
    *,
    counts_seed: int,
    train_seeds: Optional[List[int]],
    val_seeds: Optional[List[int]],
    test_seeds: Optional[List[int]],
    integrated: bool,
    mpi_world_size: int,
) -> Dict[str, Any]:
    n = diag.counts.astype(int)
    total = int(diag.total_samples)
    frac = (n / total).tolist()
    return {
        "seed": int(seed),
        "counts_seed": int(counts_seed),
        "target_counts": target_counts.astype(int).tolist(),
        "stochastic_species_split": {
            "workers": int(n[0]),
            "managers": int(n[1]),
            "ceos": int(n[2]),
            "total": total,
            "fractions_workers_managers_ceos": frac,
        },
        "distribution_layers": {
            "t": float(diag.t),
            "concentration": float(diag.concentration),
            "p_interp_even_to_target": diag.p_interp.astype(float).tolist(),
            "p_sampled_dirichlet": diag.p_sampled.astype(float).tolist(),
        },
        "dataset_seed_splits": {
            "train_seeds": list(train_seeds) if train_seeds is not None else None,
            "val_seeds": list(val_seeds) if val_seeds is not None else None,
            "test_seeds": list(test_seeds) if test_seeds is not None else None,
        },
        "trajectories": {
            "integrated_in_simulate_distribution": bool(integrated),
            "mpi_world_size": int(mpi_world_size),
            "cohort_sizes_used": [int(n[0]), int(n[1]), int(n[2])],
            "paths_relative_to_distribution_root": _trajectory_paths_relative(seed)
            if integrated
            else [],
            "note": (
                "Opinion .npy files live under <distribution_root>/data/ and match "
                "sampled_species_counts (workers, managers, CEOs)."
            ),
        },
    }


def save_distribution_run_artifacts(
    metadata_dir: Union[str, Path],
    seed: int,
    diag: SpeciesAllocationDiagnostics,
    target_counts: np.ndarray,
    counts_seed: int,
    *,
    train_seeds: Optional[List[int]] = None,
    val_seeds: Optional[List[int]] = None,
    test_seeds: Optional[List[int]] = None,
    integrated: bool = False,
    mpi_world_size: int = 1,
) -> Path:
    meta_dir = Path(metadata_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta = build_distribution_metadata(
        seed,
        diag,
        target_counts,
        counts_seed=counts_seed,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        test_seeds=test_seeds,
        integrated=integrated,
        mpi_world_size=mpi_world_size,
    )
    json_path = meta_dir / f"distribution_meta_mpi_{seed}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    counts_path = meta_dir / f"sampled_species_counts_mpi_{seed}.npy"
    np.save(counts_path, diag.counts)
    return json_path


def run_distribution_pipeline(
    seed: int,
    t: float,
    concentration: float,
    target_counts: Optional[ArrayLike] = None,
    *,
    total_samples: Optional[int] = None,
    counts_seed: Optional[int] = None,
    distribution_root: Union[str, Path] = DATA_DISTRIBUTION_DIR,
    train_seeds: Optional[List[int]] = None,
    val_seeds: Optional[List[int]] = None,
    test_seeds: Optional[List[int]] = None,
    run_integration: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, SpeciesAllocationDiagnostics, Path]:
    """
    Sample cohort sizes, optionally run inlined MPI dynamics, write data/ + metadata/ under
    distribution_root. Launch with: mpirun -np P python simulate_distribution.py
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    root = Path(distribution_root)
    data_dir = distribution_data_dir(root)
    meta_dir = distribution_metadata_dir(root)

    target = (
        np.asarray(target_counts, dtype=np.int64).ravel()
        if target_counts is not None
        else DEFAULT_TARGET_COUNTS.copy()
    )
    total = int(total_samples) if total_samples is not None else int(target.sum())
    if int(target.sum()) != total:
        raise ValueError("target_counts must sum to total_samples.")

    cs = int(counts_seed) if counts_seed is not None else int(seed) * 100_000 + 17
    rng = np.random.default_rng(cs)
    counts, diag = sample_species_counts_with_diagnostics(
        t,
        target,
        total,
        concentration,
        rng=rng,
        verbose=verbose and rank == 0,
    )

    integrated = False
    if run_integration:
        run_opinion_mpi_integration(
            seed,
            int(counts[0]),
            int(counts[1]),
            int(counts[2]),
            data_dir,
            verbose=verbose and rank == 0,
        )
        integrated = True

    json_path = meta_dir / f"distribution_meta_mpi_{seed}.json"
    if rank == 0:
        json_path = save_distribution_run_artifacts(
            meta_dir,
            seed,
            diag,
            target,
            counts_seed=cs,
            train_seeds=train_seeds,
            val_seeds=val_seeds,
            test_seeds=test_seeds,
            integrated=integrated,
            mpi_world_size=mpi_size,
        )
        if verbose:
            print(f"[simulate_distribution] saved metadata: {json_path}", flush=True)

    comm.Barrier()
    return counts, diag, json_path


def run_single_distribution_generation(
    seed: int,
    *,
    t_imbalance: float = 0.5,
    concentration: float = 50.0,
    counts_seed: Optional[int] = None,
    target_counts: Optional[ArrayLike] = None,
    total_samples: Optional[int] = None,
    distribution_root: Union[str, Path] = DATA_DISTRIBUTION_DIR,
    train_seeds: Optional[List[int]] = None,
    val_seeds: Optional[List[int]] = None,
    test_seeds: Optional[List[int]] = None,
    run_integration: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, SpeciesAllocationDiagnostics, Path]:
    tc = (
        np.asarray(target_counts, dtype=np.int64).ravel()
        if target_counts is not None
        else DEFAULT_TARGET_COUNTS.copy()
    )
    return run_distribution_pipeline(
        seed,
        t_imbalance,
        concentration,
        target_counts=tc,
        total_samples=total_samples,
        counts_seed=counts_seed,
        distribution_root=distribution_root,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        test_seeds=test_seeds,
        run_integration=run_integration,
        verbose=verbose,
    )


# --------------------- main -------------------------------------------------------------------
def main() -> None:
    SEEDS = list(range(3, 10))

    T_IMBALANCE = 0.5
    CONCENTRATION = 50.0
    COUNTS_SEED = None
    TARGET_COUNTS = DEFAULT_TARGET_COUNTS.copy()
    TOTAL_SAMPLES = None

    RUN_INTEGRATION = True
    DISTRIBUTION_ROOT = DATA_DISTRIBUTION_DIR

    TRAIN_SEEDS = list(range(0, 70))
    VAL_SEEDS = list(range(70, 85))
    TEST_SEEDS = list(range(85, 100))

    VERBOSE = True

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    for seed in SEEDS:
        if VERBOSE and rank == 0:
            print(
                f"\n{'='*60}\n[simulate_distribution] seed={seed}\n{'='*60}",
                flush=True,
            )
        run_single_distribution_generation(
            seed,
            t_imbalance=T_IMBALANCE,
            concentration=CONCENTRATION,
            counts_seed=COUNTS_SEED,
            target_counts=TARGET_COUNTS,
            total_samples=TOTAL_SAMPLES,
            distribution_root=DISTRIBUTION_ROOT,
            train_seeds=TRAIN_SEEDS,
            val_seeds=VAL_SEEDS,
            test_seeds=TEST_SEEDS,
            run_integration=RUN_INTEGRATION,
            verbose=VERBOSE,
        )


if __name__ == "__main__":
    main()
