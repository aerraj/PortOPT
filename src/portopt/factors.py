from __future__ import annotations

import numpy as np
import pandas as pd


def _cross_sectional_zscore(frame: pd.DataFrame, winsor: float = 3.0) -> pd.DataFrame:
    center = frame.mean(axis=1)
    scale = frame.std(axis=1).replace(0, np.nan)
    return frame.sub(center, axis=0).div(scale, axis=0).clip(-winsor, winsor)


def compute_factors(prices: pd.DataFrame, benchmark: pd.Series, cfg: dict) -> dict[str, pd.DataFrame]:
    """Compute point-in-time, price-derived cross-sectional factor scores."""
    returns = prices.pct_change(fill_method=None)
    vol_lb = int(cfg["volatility_lookback"])
    beta_lb = int(cfg["beta_lookback"])
    quality_lb = int(cfg["quality_lookback"])
    mom_lb = int(cfg["momentum_lookback"])
    skip = int(cfg["momentum_skip"])
    value_lb = int(cfg["value_lookback"])

    momentum = prices.shift(skip).div(prices.shift(mom_lb)).sub(1.0)
    growth = prices.shift(skip).div(prices.shift(126)).sub(1.0) - momentum / 2.0
    volatility = returns.rolling(vol_lb).std() * np.sqrt(252)

    market = benchmark.pct_change(fill_method=None)
    beta = returns.rolling(beta_lb).cov(market).div(market.rolling(beta_lb).var(), axis=0)
    low_risk = -(volatility.rank(axis=1, pct=True) + beta.abs().rank(axis=1, pct=True))

    downside = returns.clip(upper=0).rolling(quality_lb).std()
    hit_rate = (returns > 0).rolling(quality_lb).mean()
    quality = hit_rate - downside.rank(axis=1, pct=True)

    log_price = np.log(prices)
    trend = log_price.ewm(span=value_lb, min_periods=value_lb // 2, adjust=False).mean()
    value = -(log_price - trend).div(log_price.rolling(value_lb).std())

    raw = {
        "momentum": momentum,
        "value": value,
        "quality": quality,
        "low_risk": low_risk,
        "growth": growth,
    }
    return {name: _cross_sectional_zscore(signal) for name, signal in raw.items()}


REGIME_TILTS = {
    "Expansion": {"momentum": 1.35, "growth": 1.35, "value": 0.65, "quality": 0.75, "low_risk": 0.60},
    "Slowdown": {"momentum": 0.85, "growth": 0.60, "value": 0.85, "quality": 1.30, "low_risk": 1.40},
    "Contraction": {"momentum": 0.70, "growth": 0.50, "value": 1.05, "quality": 1.40, "low_risk": 1.50},
    "Recovery": {"momentum": 1.25, "growth": 1.20, "value": 1.35, "quality": 0.70, "low_risk": 0.60},
}


def composite_score(
    factors: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    regime: str,
    base_weights: dict[str, float],
) -> pd.Series:
    tilt = REGIME_TILTS.get(regime, {name: 1.0 for name in factors})
    weighted = []
    total_weight = 0.0
    for name, frame in factors.items():
        weight = float(base_weights[name]) * float(tilt[name])
        weighted.append(frame.loc[date] * weight)
        total_weight += weight
    return pd.concat(weighted, axis=1).sum(axis=1, min_count=len(weighted)) / total_weight

