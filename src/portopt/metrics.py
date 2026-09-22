from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(returns: pd.Series, equity: pd.Series) -> dict[str, float]:
    clean = returns.dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    annual_return = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0
    annual_vol = clean.std() * np.sqrt(252)
    sharpe = clean.mean() / clean.std() * np.sqrt(252) if clean.std() else np.nan
    downside = clean[clean < 0].std() * np.sqrt(252)
    sortino = clean.mean() * 252 / downside if downside else np.nan
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = drawdown.min()
    calmar = annual_return / abs(max_drawdown) if max_drawdown else np.nan
    return {
        "total_return": float(total_return),
        "cagr": float(annual_return),
        "annual_volatility": float(annual_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_drawdown),
        "calmar": float(calmar),
        "positive_days": float((clean > 0).mean()),
    }

