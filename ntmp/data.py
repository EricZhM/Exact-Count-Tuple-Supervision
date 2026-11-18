"""Offline exact-count construction and label-free NTMP training views."""
from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets


DATASETS = {"mnist", "fashionmnist", "cifar10", "svhn"}


def canonical_name(name: str) -> str:
    key = name.lower().replace("-", "")
    if key not in DATASETS:
        raise ValueError(f"unsupported dataset {name!r}; choose {sorted(DATASETS)}")
    return key


def load_official(name: str, root: str | Path, *, train: bool, download: bool = False):
    key = canonical_name(name)
    kwargs = {"root": str(root), "download": download}
    if key == "mnist":
        return datasets.MNIST(train=train, **kwargs)
    if key == "fashionmnist":
        return datasets.FashionMNIST(train=train, **kwargs)
    if key == "cifar10":
        return datasets.CIFAR10(train=train, **kwargs)
    return datasets.SVHN(split="train" if train else "test", **kwargs)


def raw_targets(raw: Any) -> np.ndarray:
    values = raw.labels if hasattr(raw, "labels") else raw.targets
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.int64)


def binary_labels(raw: Any) -> np.ndarray:
    """Original project mapping: odd original class IDs are positive."""
    return np.where(raw_targets(raw) % 2 == 1, 1, -1).astype(np.int8)


def _digest(payload: Any) -> str:
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode()).hexdigest()[:12]


def _tuple_patterns(n: int, m: int, count: int, rng: np.random.Generator) -> list[tuple[int, ...]]:
    """Balance all position patterns when feasible, as in the original code."""
    pattern_count = math.comb(n, m)
    if pattern_count <= 100_000:
        patterns = []
        for positions in combinations(range(n), m):
            row = [0] * n
            for position in positions:
                row[position] = 1
            patterns.append(tuple(row))
        repeats, remainder = divmod(count, len(patterns))
        chosen = patterns * repeats + patterns[:remainder]
        rng.shuffle(chosen)
        return chosen

    # A scalable exact-count fallback for unusually large n. It changes only
    # position allocation, which is irrelevant to the pointwise NTMP risk.
    chosen = []
    for _ in range(count):
        positions = set(rng.choice(n, size=m, replace=False).tolist())
        chosen.append(tuple(int(i in positions) for i in range(n)))
    return chosen


def prepare_manifest(
    *,
    dataset: str,
    data_dir: str | Path,
    output_dir: str | Path,
    seed: int,
    n: int,
    m: int,
    pi: float,
    tuple_count: int,
    reference_size: int,
    download: bool = False,
) -> Path:
    """Use labels once, offline, to create disjoint D_T and D_U indices."""
    dataset = canonical_name(dataset)
    if n <= 0 or not 0 <= m <= n:
        raise ValueError("require n > 0 and 0 <= m <= n")
    if not 0.0 <= pi <= 1.0:
        raise ValueError("pi must be in [0, 1]")
    if abs(pi - m / n) <= 1e-12:
        raise ValueError("pi must differ from m/n")
    if tuple_count <= 0 or reference_size <= 0:
        raise ValueError("tuple_count and reference_size must be positive")

    config = {
        "dataset": dataset,
        "seed": int(seed),
        "n": int(n),
        "m": int(m),
        "alpha": m / n,
        "pi": float(pi),
        "tuple_count": int(tuple_count),
        "reference_size": int(reference_size),
        "class_mapping": "odd_positive_even_negative",
        "sampling": "static_without_replacement",
    }
    manifest_id = _digest({**config, "data_dir": str(Path(data_dir))})
    manifest_dir = Path(output_dir) / "manifests" / f"{dataset}_{manifest_id}"
    manifest_path = manifest_dir / "data_manifest.json"
    if manifest_path.exists():
        return manifest_path

    raw = load_official(dataset, data_dir, train=True, download=download)
    labels = binary_labels(raw)
    rng = np.random.default_rng(seed)
    positives = rng.permutation(np.flatnonzero(labels == 1))
    negatives = rng.permutation(np.flatnonzero(labels == -1))

    expected_reference_positive = reference_size * pi
    reference_positive = int(round(expected_reference_positive))
    if abs(expected_reference_positive - reference_positive) > 1e-9:
        raise ValueError(
            "reference_size * pi must be an integer so D_U has exactly the requested prior"
        )
    reference_negative = reference_size - reference_positive
    needed_positive = tuple_count * m + reference_positive
    needed_negative = tuple_count * (n - m) + reference_negative
    if needed_positive > len(positives) or needed_negative > len(negatives):
        raise ValueError(
            "insufficient class examples without replacement: "
            f"need (+{needed_positive}, -{needed_negative}), "
            f"have (+{len(positives)}, -{len(negatives)})"
        )

    patterns = _tuple_patterns(n, m, tuple_count, rng)
    positive_pointer = negative_pointer = 0
    rows: list[list[int]] = []
    for pattern in patterns:
        row = []
        for is_positive in pattern:
            if is_positive:
                row.append(int(positives[positive_pointer]))
                positive_pointer += 1
            else:
                row.append(int(negatives[negative_pointer]))
                negative_pointer += 1
        rows.append(row)

    reference = np.concatenate(
        (
            positives[positive_pointer:positive_pointer + reference_positive],
            negatives[negative_pointer:negative_pointer + reference_negative],
        )
    )
    reference = rng.permutation(reference).astype(int).tolist()

    # These assertions are offline only. Labels are not stored in training views.
    assert all(int((labels[row] == 1).sum()) == m for row in rows)
    assert not ({index for row in rows for index in row} & set(reference))
    assert int((labels[reference] == 1).sum()) == reference_positive

    manifest = {
        "schema_version": 1,
        "config": config,
        "data_dir": str(Path(data_dir)),
        "sources": {
            "D_T": {"tuples": rows, "n": n, "m": m, "alpha": m / n},
            "D_U": {
                "indices": reference,
                "pi_requested": float(pi),
                "pi_realized": reference_positive / reference_size,
                "positive_count": reference_positive,
                "negative_count": reference_negative,
            },
        },
    }
    manifest_dir.mkdir(parents=True, exist_ok=False)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _image(raw: Any, index: int) -> torch.Tensor:
    image = torch.as_tensor(raw.data[index])
    if image.ndim == 2:
        image = image.unsqueeze(0)
    elif image.ndim == 3 and image.shape[-1] in (1, 3):
        image = image.permute(2, 0, 1)
    return image.float().div_(255.0)


class TupleView(Dataset):
    """Training interface returning complete tuples and no instance labels."""
    def __init__(self, raw: Any, rows: Iterable[Iterable[int]]) -> None:
        self._raw = raw
        self._rows = tuple(tuple(map(int, row)) for row in rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.stack([_image(self._raw, item) for item in self._rows[index]])


class ReferenceView(Dataset):
    """Training interface returning reference features and no instance labels."""
    def __init__(self, raw: Any, indices: Iterable[int]) -> None:
        self._raw = raw
        self._indices = tuple(map(int, indices))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        return _image(self._raw, self._indices[index])


class TestView(Dataset):
    """Official test split; labels are exposed for evaluation only."""
    def __init__(self, raw: Any) -> None:
        self._raw = raw
        self._labels = binary_labels(raw)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return _image(self._raw, index), int(self._labels[index])


def training_views(manifest_path: str | Path) -> tuple[TupleView, ReferenceView]:
    manifest = load_manifest(manifest_path)
    raw = load_official(manifest["config"]["dataset"], manifest["data_dir"], train=True)
    return (
        TupleView(raw, manifest["sources"]["D_T"]["tuples"]),
        ReferenceView(raw, manifest["sources"]["D_U"]["indices"]),
    )


def test_view(manifest_path: str | Path) -> TestView:
    manifest = load_manifest(manifest_path)
    raw = load_official(manifest["config"]["dataset"], manifest["data_dir"], train=False)
    return TestView(raw)
