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


def factor_information_coefficients(
    factors: dict[str, pd.DataFrame],
    rebalance_dates: pd.DatetimeIndex,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate each signal's next-month rank information coefficient.

    The row indexed by date t uses factor values at t and the return from t to
    the following rebalance. A strategy running at t can therefore use rows
    strictly before t, including the result of the immediately prior signal.
    """
    month_end_prices = prices.reindex(rebalance_dates)
    forward_returns = month_end_prices.pct_change(fill_method=None).shift(-1)
    history = pd.DataFrame(index=rebalance_dates[:-1], columns=factors, dtype=float)
    for signal_date in history.index:
        realized = forward_returns.loc[signal_date]
        for name, frame in factors.items():
            pair = pd.concat(
                [frame.loc[signal_date].rename("score"), realized.rename("return")], axis=1
            ).dropna()
            if len(pair) >= 10:
                history.loc[signal_date, name] = pair["score"].corr(
                    pair["return"], method="spearman"
                )
    return history


def walk_forward_factor_weights(
    ic_history: pd.DataFrame,
    regime_history: pd.Series,
    signal_date: pd.Timestamp,
    current_regime: str,
    cfg: dict,
) -> pd.Series:
    """Estimate non-negative factor weights using only previously realized ICs.

    A stable momentum core limits estimation error. The satellite allocation
    blends overall trailing evidence with evidence from prior occurrences of
    the current QMI regime, then excludes factors with non-positive evidence.
    """
    available = ic_history.loc[ic_history.index < signal_date].tail(
        int(cfg["factor_ic_lookback"])
    )
    names = ic_history.columns
    core_factor = str(cfg["core_factor"])
    core_weight = float(cfg["core_weight"])
    if available.empty:
        return pd.Series({name: float(name == core_factor) for name in names})

    halflife = float(cfg["factor_ic_halflife"])
    overall = available.ewm(halflife=halflife, min_periods=3).mean().iloc[-1]
    past_regimes = regime_history.reindex(available.index)
    regime_sample = available.loc[past_regimes == current_regime]
    if len(regime_sample) >= int(cfg["minimum_regime_observations"]):
        regime_estimate = regime_sample.ewm(halflife=halflife, min_periods=3).mean().iloc[-1]
        blend = float(cfg["regime_ic_weight"])
        evidence = (1.0 - blend) * overall + blend * regime_estimate
    else:
        evidence = overall

    satellite = evidence.clip(lower=0.0).fillna(0.0)
    if satellite.sum() == 0:
        satellite.loc[core_factor] = 1.0
    satellite /= satellite.sum()
    weights = satellite * (1.0 - core_weight)
    weights.loc[core_factor] += core_weight
    return weights / weights.sum()
