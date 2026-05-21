import os
import numpy as np
import re
import torch
from torch.utils.data import Dataset
from utils.io import load_simulation_raw


class SimulationDataset(Dataset):

    """
    Cached graph dataset.

    Each dataset item is already a PyG graph.
    """

    def __init__(
        self,
        data_dir,
        graph_builder,
        stride=1,
        random_permutation=True,
        seed=0,
    ):

        self.data_dir = data_dir

        self.graph_builder = graph_builder

        self.stride = stride

        self.random_permutation = random_permutation

        self.rng = np.random.default_rng(seed)

        # loaded raw simulations
        self.simulations = []

        # list of (sim_id, timestep)
        self.samples = []

        # cached graphs
        self.graphs = []

        # -----------------------------------------
        # pipeline
        # -----------------------------------------

        self._load_simulations()

        self._build_index()

        self._precompute_graphs()

    # ---------------------------------------------------------
    # LOAD SIMULATIONS
    # ---------------------------------------------------------
    def _load_simulations(self):
        # Discover unique seeds numerically
        seeds = []
        for f in os.listdir(self.data_dir):
            if f.startswith("opinion_workers_mpi_") and f.endswith(".npy"):
                match = re.search(r'opinion_workers_mpi_(\d+)\.npy$', f)
                if match:
                    seeds.append(int(match.group(1)))
        seeds = sorted(seeds)
        for seed in seeds:
            w, m, c = load_simulation_raw(self.data_dir, seed)
            self.simulations.append({
                "workers": w,
                "managers": m,
                "ceos": c
            })
    # ---------------------------------------------------------
    # BUILD SAMPLE INDEX
    # ---------------------------------------------------------

    def _build_index(self):

        for sim_id, sim in enumerate(self.simulations):

            T = sim["workers"].shape[0]

            for t in range(0, T - 1, self.stride):

                self.samples.append((sim_id, t))

    # ---------------------------------------------------------
    # RAW SAMPLE
    # ---------------------------------------------------------

    def _get_raw_sample(self, idx):

        sim_id, t = self.samples[idx]

        sim = self.simulations[sim_id]

        workers = sim["workers"]
        managers = sim["managers"]
        ceos = sim["ceos"]

        # -----------------------------------------
        # timestep t
        # -----------------------------------------

        x1_t = workers[t]
        x2_t = managers[t]
        x3_t = ceos[t]

        # -----------------------------------------
        # timestep t+1
        # -----------------------------------------

        x1_t1 = workers[t + 1]
        x2_t1 = managers[t + 1]
        x3_t1 = ceos[t + 1]

        # -----------------------------------------
        # merge species
        # -----------------------------------------

        x_t = np.concatenate([
            x1_t,
            x2_t,
            x3_t,
        ])

        x_t1 = np.concatenate([
            x1_t1,
            x2_t1,
            x3_t1,
        ])

        # -----------------------------------------
        # labels
        # -----------------------------------------

        labels = np.concatenate([

            np.zeros(len(x1_t), dtype=np.int64),

            np.ones(len(x2_t), dtype=np.int64),

            2 * np.ones(len(x3_t), dtype=np.int64),
        ])

        # -----------------------------------------
        # random permutation
        # -----------------------------------------

        if self.random_permutation:

            perm = self.rng.permutation(len(x_t))

            x_t = x_t[perm]
            x_t1 = x_t1[perm]
            labels = labels[perm]

        # -----------------------------------------
        # tensors
        # -----------------------------------------

        sample = {

            "x_t": torch.tensor(
                x_t,
                dtype=torch.float32
            ),

            "x_t1": torch.tensor(
                x_t1,
                dtype=torch.float32
            ),

            "labels": torch.tensor(
                labels,
                dtype=torch.long
            ),

            "sim_id": sim_id,

            "timestep": t,
        }

        return sample

    # ---------------------------------------------------------
    # PRECOMPUTE GRAPHS
    # ---------------------------------------------------------

    def _precompute_graphs(self):

        print("\nPrecomputing graphs...\n")

        for idx in range(len(self.samples)):

            raw_sample = self._get_raw_sample(idx)

            graph = self.graph_builder.build_graph(
                raw_sample
            )

            self.graphs.append(graph)

            if (idx + 1) % 10 == 0:

                print(
                    f"Built {idx+1}/{len(self.samples)} graphs"
                )

        print("\nFinished graph precomputation.\n")

    # ---------------------------------------------------------
    # DATASET SIZE
    # ---------------------------------------------------------

    def __len__(self):

        return len(self.graphs)

    # ---------------------------------------------------------
    # RETURN GRAPH
    # ---------------------------------------------------------

    def __getitem__(self, idx):

        return self.graphs[idx]