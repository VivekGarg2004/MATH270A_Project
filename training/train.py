import torch
import torch.nn.functional as F

from training.metrics import compute_metrics


def train_one_epoch(
    model,
    dataset,
    optimizer,
    device,
    class_weights=None,  # <-- Add this parameter
):
    model.train()
    total_loss = 0.0

    for i, graph in enumerate(dataset):
        if i % 100 == 0:
            print(f"Processing sample {i}/{len(dataset)}")

        graph = graph.to(device)
        optimizer.zero_grad()

        logits = model(graph)

        # Pass the weights into cross_entropy
        loss = F.cross_entropy(
            logits,
            graph.y,
            weight=class_weights  # <-- Apply the weights here
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    total_loss /= len(dataset)
    return total_loss


def evaluate(
    model,
    dataset,
    device,
):

    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():

        for graph in dataset:

            graph = graph.to(device)

            logits = model(graph)

            preds = logits.argmax(dim=1)

            all_preds.append(
                preds.cpu()
            )

            all_targets.append(
                graph.y.cpu()
            )

    all_preds = torch.cat(all_preds).numpy()

    all_targets = torch.cat(all_targets).numpy()

    metrics = compute_metrics(
        all_targets,
        all_preds
    )

    return metrics