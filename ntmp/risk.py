"""NTMP risk estimators from the paper equations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


Correction = Literal["ure", "abs", "relu"]


@dataclass(frozen=True)
class RiskResult:
    objective: torch.Tensor
    r_positive: torch.Tensor
    r_negative: torch.Tensor
    e_t_positive: torch.Tensor
    e_t_negative: torch.Tensor
    e_u_positive: torch.Tensor
    e_u_negative: torch.Tensor


def point_losses(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Original bounded losses h_+(x)=sigmoid(-g(x)), h_-(x)=sigmoid(g(x))."""
    scores = logits.reshape(-1)
    return torch.sigmoid(-scores), torch.sigmoid(scores)


def risk_from_expectations(
    e_t_positive: torch.Tensor,
    e_t_negative: torch.Tensor,
    e_u_positive: torch.Tensor,
    e_u_negative: torch.Tensor,
    *,
    alpha: float,
    pi: float,
    correction: Correction = "ure",
) -> RiskResult:
    """Reconstruct the positive and negative class-risk components.

    ``alpha=m/n`` is the point-level positive proportion in D_T and ``pi`` is
    the positive prior of D_U.  T/U ordering follows the paper exactly.
    """
    if not (0.0 <= alpha <= 1.0 and 0.0 <= pi <= 1.0):
        raise ValueError("alpha and pi must be in [0, 1]")
    if abs(pi - alpha) <= 1e-12:
        raise ValueError("NTMP is not identifiable when pi == m/n")

    denominator = pi - alpha
    r_positive = pi / denominator * (
        (1.0 - alpha) * e_u_positive - (1.0 - pi) * e_t_positive
    )
    r_negative = (1.0 - pi) / denominator * (
        -alpha * e_u_negative + pi * e_t_negative
    )

    if correction == "ure":
        objective = r_positive + r_negative
    elif correction == "abs":
        objective = r_positive.abs() + r_negative.abs()
    elif correction == "relu":
        objective = torch.relu(r_positive) + torch.relu(r_negative)
    else:
        raise ValueError(f"unknown correction: {correction}")

    return RiskResult(
        objective=objective,
        r_positive=r_positive,
        r_negative=r_negative,
        e_t_positive=e_t_positive,
        e_t_negative=e_t_negative,
        e_u_positive=e_u_positive,
        e_u_negative=e_u_negative,
    )


def ntmp_risk(
    tuple_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    *,
    n: int,
    m: int,
    pi: float,
    correction: Correction = "ure",
) -> RiskResult:
    if n <= 0 or not 0 <= m <= n:
        raise ValueError("require n > 0 and 0 <= m <= n")
    t_positive, t_negative = point_losses(tuple_logits)
    u_positive, u_negative = point_losses(reference_logits)
    return risk_from_expectations(
        t_positive.mean(),
        t_negative.mean(),
        u_positive.mean(),
        u_negative.mean(),
        alpha=m / n,
        pi=pi,
        correction=correction,
    )
