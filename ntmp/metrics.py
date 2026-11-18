"""Dependency-light binary classification metrics."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def metrics_from_scores(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = (np.asarray(labels).reshape(-1) == 1).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    predicted = (scores >= 0.5).astype(np.int64)
    accuracy = float((predicted == labels).mean())

    f1_scores = []
    for class_id in (0, 1):
        true_positive = int(((predicted == class_id) & (labels == class_id)).sum())
        false_positive = int(((predicted == class_id) & (labels != class_id)).sum())
        false_negative = int(((predicted != class_id) & (labels == class_id)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)

    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUROC requires both classes")
    ranks = _average_ranks(scores)
    positive_rank_sum = float(ranks[labels == 1].sum())
    auroc = (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)
    return {"accuracy": accuracy, "macro_f1": float(np.mean(f1_scores)), "auroc": float(auroc)}


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataset, *, device: str, batch_size: int = 512):
    model.eval()
    labels, scores = [], []
    for features, target in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        score = torch.sigmoid(model(features.to(device)).reshape(-1))
        scores.append(score.cpu())
        labels.append(target)
    return metrics_from_scores(torch.cat(labels).numpy(), torch.cat(scores).numpy())
