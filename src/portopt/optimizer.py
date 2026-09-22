from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _constraint_functions(sectors: pd.Series, cfg: dict):
    gross = float(cfg["gross_exposure"])
    net_limit = float(cfg["net_exposure_limit"])
    sector_gross = float(cfg["max_sector_gross"])
    sector_net = float(cfg["max_sector_net"])

    constraints = [
        {"type": "ineq", "fun": lambda w: gross - np.abs(w).sum()},
        {"type": "ineq", "fun": lambda w: net_limit - abs(w.sum())},
    ]
    for sector in sorted(sectors.unique()):
        mask = (sectors.to_numpy() == sector).astype(float)
        constraints.extend(
            [
                {"type": "ineq", "fun": lambda w, m=mask: sector_gross - np.abs(w * m).sum()},
                {"type": "ineq", "fun": lambda w, m=mask: sector_net - abs((w * m).sum())},
            ]
        )
    return constraints


def optimize_weights(
    score: pd.Series,
    covariance: pd.DataFrame,
    sectors: pd.Series,
    previous: pd.Series | None,
    cfg: dict,
    tail_fraction: float,
) -> pd.Series:
    """Solve a constrained long/short mean-variance allocation with SLSQP."""
    common = score.dropna().index.intersection(covariance.dropna(how="all").index)
    if len(common) < 10:
        raise ValueError("Insufficient eligible securities for optimization")
    score = score.loc[common]
    covariance = covariance.loc[common, common].fillna(0.0)
    sectors = sectors.loc[common]
    previous = pd.Series(0.0, index=common) if previous is None else previous.reindex(common).fillna(0.0)

    cutoff_low = score.quantile(tail_fraction)
    cutoff_high = score.quantile(1.0 - tail_fraction)
    bounds = []
    cap = float(cfg["max_stock_weight"])
    for value in score:
        if value >= cutoff_high:
            bounds.append((0.0, cap))
        elif value <= cutoff_low:
            bounds.append((-cap, 0.0))
        else:
            bounds.append((0.0, 0.0))

    sigma = covariance.to_numpy() * 252.0
    alpha = score.to_numpy()
    alpha /= np.linalg.norm(alpha) or 1.0
    prev = previous.to_numpy()
    risk_aversion = float(cfg["risk_aversion"])
    turnover_penalty = float(cfg["turnover_penalty"])

    def objective(w: np.ndarray) -> float:
        smooth_turnover = np.sqrt((w - prev) ** 2 + 1e-8).sum()
        return -(alpha @ w) + risk_aversion * (w @ sigma @ w) + turnover_penalty * smooth_turnover

    n_long = sum(lo >= 0 and hi > 0 for lo, hi in bounds)
    n_short = sum(lo < 0 and hi <= 0 for lo, hi in bounds)
    x0 = np.zeros(len(common))
    if n_long:
        x0[[i for i, (_, hi) in enumerate(bounds) if hi > 0]] = min(0.45 / n_long, cap)
    if n_short:
        x0[[i for i, (lo, _) in enumerate(bounds) if lo < 0]] = -min(0.45 / n_short, cap)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=_constraint_functions(sectors, cfg),
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    weights = pd.Series(result.x, index=common)
    weights[weights.abs() < 1e-7] = 0.0
    return weights

