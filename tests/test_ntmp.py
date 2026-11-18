from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

import train
from ntmp import data
from ntmp.models import MLP, ResNet20
from ntmp.risk import ntmp_risk, risk_from_expectations
from ntmp.training import train_epoch


def test_cli_arguments():
    args = train.build_parser().parse_args(
        [
            "--dataset", "mnist", "--data-dir", "data", "--n", "5", "--m", "2",
            "--pi", "0.8", "--correction", "relu", "--seed", "7",
        ]
    )
    assert (args.dataset, args.n, args.m, args.pi, args.correction, args.seed) == (
        "mnist", 5, 2, 0.8, "relu", 7
    )


@pytest.mark.parametrize("correction", ["ure", "abs", "relu"])
def test_paper_risk_formula(correction):
    values = [torch.tensor(value, dtype=torch.float64) for value in (0.2, 0.7, 0.4, 0.6)]
    result = risk_from_expectations(*values, alpha=0.4, pi=0.8, correction=correction)
    expected_positive = 0.8 / 0.4 * (0.6 * 0.4 - 0.2 * 0.2)
    expected_negative = 0.2 / 0.4 * (-0.4 * 0.6 + 0.8 * 0.7)
    assert result.r_positive.item() == pytest.approx(expected_positive)
    assert result.r_negative.item() == pytest.approx(expected_negative)
    expected = {
        "ure": expected_positive + expected_negative,
        "abs": abs(expected_positive) + abs(expected_negative),
        "relu": max(0.0, expected_positive) + max(0.0, expected_negative),
    }[correction]
    assert result.objective.item() == pytest.approx(expected)


class FakeRaw:
    def __init__(self):
        self.data = torch.arange(100 * 28 * 28, dtype=torch.uint8).reshape(100, 28, 28)
        self.targets = torch.arange(100) % 2


def test_exact_count_manifest_and_label_free_views(monkeypatch):
    workspace = Path(".test_workspace")
    raw = FakeRaw()
    monkeypatch.setattr(data, "load_official", lambda *args, **kwargs: raw)
    manifest_path = data.prepare_manifest(
        dataset="mnist", data_dir=workspace, output_dir=workspace, seed=3,
        n=5, m=2, pi=0.8, tuple_count=4, reference_size=10,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels = data.binary_labels(raw)
    rows = manifest["sources"]["D_T"]["tuples"]
    reference = manifest["sources"]["D_U"]["indices"]
    assert all(int((labels[row] == 1).sum()) == 2 for row in rows)
    assert int((labels[reference] == 1).sum()) == 8
    assert not ({item for row in rows for item in row} & set(reference))
    tuple_view, reference_view = data.training_views(manifest_path)
    assert not hasattr(tuple_view, "targets") and not hasattr(reference_view, "targets")


@pytest.mark.parametrize(
    ("model", "features"),
    [
        (MLP(), torch.rand(4, 1, 28, 28)),
        (ResNet20(), torch.rand(2, 3, 32, 32)),
    ],
)
def test_models_forward_backward(model, features):
    output = model(features)
    assert output.shape == (len(features), 1)
    output.mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_minimal_training_flow():
    torch.manual_seed(1)
    tuples = TensorDataset(torch.rand(4, 2, 1, 2, 2))
    reference = TensorDataset(torch.rand(4, 1, 2, 2))

    class FeaturesOnly:
        def __init__(self, wrapped):
            self.wrapped = wrapped
        def __len__(self):
            return len(self.wrapped)
        def __getitem__(self, index):
            return self.wrapped[index][0]

    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 1))
    before = model[1].weight.detach().clone()
    metrics = train_epoch(
        model, FeaturesOnly(tuples), FeaturesOnly(reference),
        torch.optim.Adam(model.parameters(), lr=1e-3),
        n=2, m=1, pi=0.8, correction="abs", tuple_batch_size=2,
        device="cpu", seed=11, max_batches=1,
    )
    assert metrics["updates"] == 1
    assert np.isfinite(metrics["train_loss"])
    assert not torch.equal(before, model[1].weight.detach())


def test_risk_backward_is_finite():
    tuple_logits = torch.randn(12, requires_grad=True)
    reference_logits = torch.randn(8, requires_grad=True)
    result = ntmp_risk(tuple_logits, reference_logits, n=3, m=1, pi=0.8, correction="relu")
    result.objective.backward()
    assert torch.isfinite(tuple_logits.grad).all()
    assert torch.isfinite(reference_logits.grad).all()
