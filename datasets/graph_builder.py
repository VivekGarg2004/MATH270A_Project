import torch

from torch_geometric.data import Data
from torch_geometric.nn import knn_graph



class GraphBuilder:

    def __init__(self, k_neighbors=32):

        self.k_neighbors = k_neighbors

    def build_graph(self, sample):

        """
        Input:
            sample = {
                "x_t": [N]
                "x_t1": [N]
                "labels": [N]
            }

        Output:
            PyG Data object
        """

        x_t = sample["x_t"]
        x_t1 = sample["x_t1"]
        labels = sample["labels"]


        velocity = x_t1 - x_t

        # shape: [N, 2]
        node_features = torch.stack([
            x_t,
            velocity
        ], dim=1)

        coords = x_t.unsqueeze(1)

        edge_index = knn_graph(
            coords,
            k=self.k_neighbors,
            loop=False
        )

        src = edge_index[0]
        dst = edge_index[1]

        dx = x_t[dst] - x_t[src]

        edge_attr = torch.stack([
            dx,
            torch.abs(dx)
        ], dim=1)

        graph = Data(

            x=node_features,

            edge_index=edge_index,

            edge_attr=edge_attr,

            y=labels,
        )

        return graph