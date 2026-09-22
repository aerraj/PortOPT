from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from portopt.factors import (
    composite_score,
    compute_factors,
    factor_information_coefficients,
    walk_forward_factor_weights,
)
from portopt.metrics import performance_metrics
from portopt.optimizer import optimize_weights
from portopt.regimes import classify_regimes


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    regimes: pd.Series
    factor_weights: pd.DataFrame
    metrics: dict[str, float]


def run_backtest(
    prices: pd.DataFrame,
    benchmark: pd.Series,
    qmi: pd.Series,
    sectors: pd.Series,
    config: dict,
) -> BacktestResult:
    """Run a close-to-close monthly-rebalanced backtest with one-period signal lag."""
    signal_cfg = config["signals"]
    opt_cfg = config["optimizer"]
    bt_cfg = config["backtest"]
    initial = float(config["project"]["initial_capital"])

    prices = prices.sort_index().ffill(limit=5)
    benchmark = benchmark.reindex(prices.index).ffill()
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    factors = compute_factors(prices, benchmark, signal_cfg)

    rebalances = prices.groupby(prices.index.to_period("M")).tail(1).index
    rebalances = rebalances[rebalances >= prices.index[int(signal_cfg["minimum_history"])] ]
    monthly_qmi = qmi.reindex(rebalances, method="ffill")
    regimes = classify_regimes(monthly_qmi)
    ic_history = factor_information_coefficients(factors, rebalances, prices)

    targets: dict[pd.Timestamp, pd.Series] = {}
    factor_weight_history: dict[pd.Timestamp, pd.Series] = {}
    previous = pd.Series(0.0, index=prices.columns)
    cov_lb = int(opt_cfg["covariance_lookback"])
    for signal_date in rebalances[:-1]:
        execution_candidates = prices.index[prices.index > signal_date]
        if execution_candidates.empty:
            break
        execution_date = execution_candidates[0]
        if signal_cfg.get("adaptive_factor_weights", False):
            factor_weights = walk_forward_factor_weights(
                ic_history,
                regimes,
                signal_date,
                regimes.loc[signal_date],
                signal_cfg,
            ).to_dict()
            regime_for_tilt = "Adaptive"
        else:
            factor_weights = signal_cfg["factor_weights"]
            regime_for_tilt = regimes.loc[signal_date]
        score = composite_score(factors, signal_date, regime_for_tilt, factor_weights)
        history = returns.loc[:signal_date].tail(cov_lb)
        eligible = history.notna().mean()[lambda x: x >= 0.95].index.intersection(score.dropna().index)
        if len(eligible) < 10:
            continue
        covariance = history[eligible].cov()
        weights = optimize_weights(
            score.loc[eligible], covariance, sectors, previous, opt_cfg, float(signal_cfg["tail_fraction"])
        ).reindex(prices.columns).fillna(0.0)
        targets[execution_date] = weights
        factor_weight_history[signal_date] = pd.Series(factor_weights, dtype=float)
        previous = weights

    if not targets:
        raise RuntimeError("No rebalance dates produced a valid portfolio")
    weight_frame = pd.DataFrame(targets).T.reindex(prices.index).ffill().fillna(0.0)
    held_weights = weight_frame.shift(1).fillna(0.0)
    gross_return = (held_weights * returns).sum(axis=1)
    daily_turnover = weight_frame.diff().abs().sum(axis=1).fillna(weight_frame.abs().sum(axis=1))
    transaction_cost = daily_turnover * float(bt_cfg["transaction_cost_bps"]) / 10_000.0
    short_exposure = held_weights.clip(upper=0).abs().sum(axis=1)
    borrow_cost = short_exposure * float(bt_cfg["annual_short_borrow_bps"]) / 10_000.0 / 252.0
    net_return = gross_return - transaction_cost - borrow_cost
    start = min(targets)
    net_return = net_return.loc[start:]
    equity = initial * (1.0 + net_return).cumprod()
    metrics = performance_metrics(net_return, equity)
    metrics.update(
        {
            "average_monthly_turnover": float(daily_turnover.loc[start:][daily_turnover > 0].mean()),
            "average_gross_exposure": float(held_weights.loc[start:].abs().sum(axis=1).mean()),
            "average_net_exposure": float(held_weights.loc[start:].sum(axis=1).mean()),
            "final_value": float(equity.iloc[-1]),
        }
    )
    return BacktestResult(
        equity=equity.rename("portfolio_value"),
        returns=net_return.rename("net_return"),
        weights=weight_frame.loc[start:],
        turnover=daily_turnover.loc[start:].rename("turnover"),
        regimes=regimes,
        factor_weights=pd.DataFrame(factor_weight_history).T,
        metrics=metrics,
    )
