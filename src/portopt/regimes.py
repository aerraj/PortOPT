from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

REGIME_ORDER = ["Contraction", "Recovery", "Expansion", "Slowdown"]


def classify_regimes(qmi: pd.Series) -> pd.Series:
    """Classify level/direction combinations into four economic regimes."""
    change = qmi.diff()
    labels = np.select(
        [
            (qmi < 0) & (change < 0),
            (qmi < 0) & (change >= 0),
            (qmi >= 0) & (change >= 0),
            (qmi >= 0) & (change < 0),
        ],
        REGIME_ORDER,
        default="Unknown",
    )
    return pd.Series(labels, index=qmi.index, name="regime")


def estimate_ou_half_life(qmi: pd.Series) -> float:
    """Estimate OU half-life in observation periods using an AR(1) regression."""
    clean = qmi.dropna()
    lag = clean.shift(1).dropna()
    delta = clean.diff().dropna().reindex(lag.index)
    beta = np.polyfit(lag.to_numpy(), delta.to_numpy(), 1)[0]
    if not -1.0 < beta < 0.0:
        return float("nan")
    theta = -np.log1p(beta)
    return float(np.log(2.0) / theta)


def gmm_regime_diagnostic(qmi: pd.Series, n_components: int = 4) -> pd.DataFrame:
    """Fit a reproducible GMM as a diagnostic, not as the tradable signal."""
    features = pd.concat(
        [qmi.rename("level"), qmi.diff().rename("change")], axis=1
    ).dropna()
    model = GaussianMixture(n_components=n_components, random_state=42, n_init=20)
    cluster = model.fit_predict(features)
    probability = model.predict_proba(features).max(axis=1)
    return pd.DataFrame({"cluster": cluster, "probability": probability}, index=features.index)
