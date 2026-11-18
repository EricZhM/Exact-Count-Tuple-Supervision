"""Minimal public implementation of n-Tuple with m-Positive (NTMP)."""

from .risk import RiskResult, ntmp_risk, risk_from_expectations

__all__ = ["RiskResult", "ntmp_risk", "risk_from_expectations"]
