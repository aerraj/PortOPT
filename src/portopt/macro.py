from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FRED_SERIES = {
    "business_confidence": "BSCICP03EZM665S",
    "unemployment": "LRHUTTTTEZM156S",
    "inflation_yoy": "EA19CPALTT01GYM",
}

ECB_DEPOSIT_RATE_URL = (
    "https://data-api.ecb.europa.eu/service/data/"
    "FM/D.U2.EUR.4F.KR.DFR.LEV?format=csvdata"
)


def _download_fred_series(series_id: str, timeout: int = 30) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text), na_values=".")
    frame.columns = ["date", series_id]
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date")[series_id].astype(float)


def _download_ecb_deposit_rate(timeout: int = 30) -> pd.Series:
    response = requests.get(ECB_DEPOSIT_RATE_URL, timeout=timeout)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text), usecols=["TIME_PERIOD", "OBS_VALUE"])
    frame["TIME_PERIOD"] = pd.to_datetime(frame["TIME_PERIOD"])
    return frame.set_index("TIME_PERIOD")["OBS_VALUE"].astype(float).rename("deposit_rate")


def download_macro(cache_dir: str | Path, refresh: bool = False) -> pd.DataFrame:
    """Fetch publicly accessible Euro-area macro series distributed by FRED."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "euro_macro.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    series = {name: _download_fred_series(sid) for name, sid in FRED_SERIES.items()}
    series["deposit_rate"] = _download_ecb_deposit_rate()
    macro = pd.concat(series, axis=1).sort_index()
    macro.to_csv(path, float_format="%.8f")
    return macro


def _expanding_zscore(series: pd.Series, min_periods: int = 36) -> pd.Series:
    mean = series.expanding(min_periods=min_periods).mean()
    std = series.expanding(min_periods=min_periods).std().replace(0, np.nan)
    return (series - mean) / std


def build_public_qmi(macro: pd.DataFrame, release_lag_months: int = 2) -> pd.DataFrame:
    """Build a public-data QMI proxy without full-sample normalization.

    Monthly observations are shifted by a conservative publication lag before
    they are visible to the strategy. Expanding statistics avoid look-ahead.
    """
    monthly = macro.resample("ME").last().ffill()
    components = pd.DataFrame(index=monthly.index)
    components["confidence"] = _expanding_zscore(monthly["business_confidence"])
    components["labour"] = -_expanding_zscore(monthly["unemployment"].diff(3))
    components["price_stability"] = -_expanding_zscore((monthly["inflation_yoy"] - 2.0).abs())
    components["policy_impulse"] = -_expanding_zscore(monthly["deposit_rate"].diff(3))
    components = components.shift(release_lag_months)
    available = components.notna().sum(axis=1)
    qmi = components.mean(axis=1).where(available >= 3).rename("qmi")
    result = components.assign(qmi=qmi, qmi_change=qmi.diff())
    return result.dropna(subset=["qmi", "qmi_change"])
