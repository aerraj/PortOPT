from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf


def load_universe(path: str | Path) -> pd.DataFrame:
    universe = pd.read_csv(path)
    required = {"ticker", "name", "sector", "country", "currency"}
    missing = required.difference(universe.columns)
    if missing:
        raise ValueError(f"Universe is missing columns: {sorted(missing)}")
    if universe["ticker"].duplicated().any():
        raise ValueError("Universe contains duplicate tickers")
    return universe.set_index("ticker")


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        raise RuntimeError("Market data provider returned no observations")
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        if "Close" not in level0:
            raise RuntimeError("Close prices are absent from market data response")
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices.sort_index().astype(float)


def download_prices(
    tickers: list[str],
    start: str,
    end: str | None,
    cache_dir: str | Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download adjusted daily closes and cache them locally.

    Yahoo Finance data is used for research convenience. The cache makes every
    completed experiment reproducible and avoids unnecessary provider requests.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    end_label = end or datetime.now(UTC).date().isoformat()
    cache_path = cache_dir / f"prices_{start}_{end_label}.csv"
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        actions=False,
        group_by="column",
        threads=True,
        progress=False,
        timeout=30,
    )
    prices = _extract_close(raw, tickers)
    coverage = prices.notna().mean()
    valid = coverage[coverage >= 0.60].index
    prices = prices.loc[:, valid].dropna(how="all")
    if prices.shape[1] < 12:
        raise RuntimeError(f"Only {prices.shape[1]} securities passed data-quality checks")
    prices.to_csv(cache_path, float_format="%.8f")
    return prices
