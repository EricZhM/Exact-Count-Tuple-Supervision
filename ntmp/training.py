"""Original NTMP training protocol with explicit result/config persistence."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import load_manifest, test_view, training_views
from .metrics import evaluate
from .models import build_model
from .risk import ntmp_risk


LEARNING_RATES = {
    "mnist": 1e-5,
    "fashionmnist": 1e-4,
    "cifar10": 1e-4,
    "svhn": 1e-3,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")
    return requested


def train_epoch(
    model: torch.nn.Module,
    tuples,
    reference,
    optimizer: torch.optim.Optimizer,
    *,
    n: int,
    m: int,
    pi: float,
    correction: str,
    tuple_batch_size: int,
    device: str,
    seed: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Train one epoch without reading instance labels."""
    tuple_generator = torch.Generator().manual_seed(seed)
    reference_generator = torch.Generator().manual_seed(seed + 1)
    reference_batch_size = max(
        1, min(len(reference), round(len(reference) / len(tuples) * tuple_batch_size))
    )
    tuple_loader = DataLoader(
        tuples, batch_size=tuple_batch_size, shuffle=True, generator=tuple_generator
    )
    reference_loader = DataLoader(
        reference, batch_size=reference_batch_size, shuffle=True, generator=reference_generator
    )

    model.train()
    totals = np.zeros(3, dtype=np.float64)
    updates = 0
    for tuple_features, reference_features in zip(tuple_loader, reference_loader):
        if max_batches is not None and updates >= max_batches:
            break
        tuple_features = tuple_features.to(device)
        reference_features = reference_features.to(device)
        tuple_points = tuple_features.flatten(0, 1)

        # One joint forward preserves the original BatchNorm behavior.
        joint = model(torch.cat((tuple_points, reference_features), dim=0)).reshape(-1)
        result = ntmp_risk(
            joint[: len(tuple_points)],
            joint[len(tuple_points):],
            n=n,
            m=m,
            pi=pi,
            correction=correction,
        )
        if not torch.isfinite(result.objective):
            raise FloatingPointError("non-finite NTMP objective")
        optimizer.zero_grad(set_to_none=True)
        result.objective.backward()
        optimizer.step()
        totals += [
            float(result.objective.detach()),
            float(result.r_positive.detach()),
            float(result.r_negative.detach()),
        ]
        updates += 1

    if updates == 0:
        raise RuntimeError("no training batches were processed")
    averages = totals / updates
    return {
        "train_loss": float(averages[0]),
        "r_positive": float(averages[1]),
        "r_negative": float(averages[2]),
        "updates": updates,
    }


def run_training(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    correction: str,
    epochs: int,
    batch_size: int,
    device: str,
    max_batches: int | None = None,
) -> Path:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided")
    manifest = load_manifest(manifest_path)
    config = manifest["config"]
    dataset = config["dataset"]
    seed = int(config["seed"])
    device = resolve_device(device)
    run_dir = Path(output_dir) / "runs" / dataset / f"ntmp_{correction}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    saved_config = {
        **config,
        "method": f"NTMP-{correction.upper()}",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATES[dataset],
        "point_loss": "sigmoid",
        "device": device,
        "normalization": "uint8 / 255; no augmentation",
        "model_selection": "last epoch; test metrics are not used for training or selection",
        "max_batches_per_epoch": max_batches,
    }
    (run_dir / "config.json").write_text(json.dumps(saved_config, indent=2), encoding="utf-8")

    seed_everything(seed)
    tuples, reference = training_views(manifest_path)
    evaluation = test_view(manifest_path)
    model = build_model(dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATES[dataset])
    history_path = run_dir / "history.csv"
    fields = [
        "epoch", "train_loss", "r_positive", "r_negative", "updates",
        "accuracy", "macro_f1", "auroc",
    ]
    last_metrics: dict[str, float] = {}
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(epochs):
            train_metrics = train_epoch(
                model,
                tuples,
                reference,
                optimizer,
                n=config["n"],
                m=config["m"],
                pi=config["pi"],
                correction=correction,
                tuple_batch_size=batch_size,
                device=device,
                seed=seed * 1_000_003 + epoch,
                max_batches=max_batches,
            )
            last_metrics = evaluate(model, evaluation, device=device)
            row = {"epoch": epoch + 1, **train_metrics, **last_metrics}
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row, sort_keys=True), flush=True)

    result = {
        "dataset": dataset,
        "method": f"NTMP-{correction.upper()}",
        "seed": seed,
        "final_epoch": epochs,
        **last_metrics,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "config": saved_config}, run_dir / "checkpoint_last.pt")
    return run_dir
