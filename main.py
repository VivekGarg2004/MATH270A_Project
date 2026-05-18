import torch

from torch.utils.data import random_split

from datasets.simulation_dataset import SimulationDataset
from datasets.graph_builder import GraphBuilder

from models.gnn.gnn_classifier import GraphSAGEClassifier

from training.train import (
    train_one_epoch,
    evaluate,
)


# ---------------------------------------------------------
# device
# ---------------------------------------------------------

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")

print(f"\nUsing device: {device}\n")


# ---------------------------------------------------------
# dataset
# ---------------------------------------------------------

dataset = SimulationDataset(
    data_dir="data",
    graph_builder=GraphBuilder(k_neighbors=16),
    stride=50,
)

print(f"Dataset size: {len(dataset)}")


# ---------------------------------------------------------
# split
# ---------------------------------------------------------

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size]
)

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples : {len(test_dataset)}")


# ---------------------------------------------------------
# graph builder
# ---------------------------------------------------------


# ---------------------------------------------------------
# model
# ---------------------------------------------------------

model = GraphSAGEClassifier(
    in_channels=2,
    hidden_channels=32,
    num_classes=3,
    num_layers=2,
    dropout=0.1,
)

model = model.to(device)

print("\nModel:")
print(model)


# ---------------------------------------------------------
# optimizer
# ---------------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


# ---------------------------------------------------------
# training loop
# ---------------------------------------------------------

epochs = 20
class_counts = torch.tensor([3232000.0, 808000.0, 40400.0])
total_samples = class_counts.sum()
class_weights = total_samples / class_counts
class_weights = class_weights / class_weights.min()
class_weights = class_weights.to(device)

for epoch in range(epochs):

    loss = train_one_epoch(
        model,
        train_dataset,
        optimizer,
        device,
        class_weights=class_weights  # <-- Example weights
    )

    metrics = evaluate(
        model,
        test_dataset,
        device,
    )

    print("\n" + "=" * 60)

    print(f"Epoch {epoch+1:03d}")

    print(f"Loss      : {loss:.4f}")

    print(f"Macro F1  : {metrics['macro_f1']:.4f}")

    print(f"Worker F1 : {metrics['worker_f1']:.4f}")
    print(f"Manager F1: {metrics['manager_f1']:.4f}")
    print(f"CEO F1    : {metrics['ceo_f1']:.4f}")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    print("=" * 60)