import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv


class GraphSAGEClassifier(nn.Module):

    def __init__(
        self,
        in_channels=2,
        hidden_channels=32,
        num_classes=3,
        num_layers=2,
        dropout=0.1,
    ):

        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # -----------------------------------------
        # graph convolution layers
        # -----------------------------------------

        self.convs = nn.ModuleList()

        #
        # first layer
        #
        self.convs.append(
            SAGEConv(in_channels, hidden_channels)
        )

        #
        # hidden layers
        #
        for _ in range(num_layers - 1):

            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels)
            )

        # -----------------------------------------
        # classifier head
        # -----------------------------------------

        self.classifier = nn.Linear(
            hidden_channels,
            num_classes
        )

    # -------------------------------------------------
    # forward pass
    # -------------------------------------------------

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index

        # -----------------------------------------
        # message passing
        # -----------------------------------------

        for conv in self.convs:

            x = conv(x, edge_index)

            x = F.relu(x)

            x = F.dropout(
                x,
                p=self.dropout,
                training=self.training
            )

        # -----------------------------------------
        # node classification
        # -----------------------------------------

        logits = self.classifier(x)

        return logits