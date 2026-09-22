from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from portopt.backtest import BacktestResult


def write_reports(result: BacktestResult, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.equity.to_csv(output_dir / "equity_curve.csv", float_format="%.8f")
    result.returns.to_csv(output_dir / "daily_returns.csv", float_format="%.10f")
    result.weights.to_csv(output_dir / "weights.csv", float_format="%.8f")
    result.turnover.to_csv(output_dir / "turnover.csv", float_format="%.8f")
    result.regimes.to_csv(output_dir / "regimes.csv")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result.metrics, handle, indent=2, sort_keys=True)

    equity_normalized = result.equity / result.equity.iloc[0]
    drawdown = result.equity / result.equity.cummax() - 1.0
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[2, 1])
    axes[0].plot(equity_normalized.index, equity_normalized, color="#12355b", linewidth=1.8)
    axes[0].set_ylabel("Growth of $1")
    axes[0].set_title("PortOPT - QMI regime-aware long/short strategy")
    axes[0].grid(alpha=0.2)
    axes[1].fill_between(drawdown.index, drawdown, 0, color="#b23a48", alpha=0.75)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "performance.png", dpi=180)
    plt.close(fig)

