import numpy as np
import pandas as pd

from portopt.regimes import classify_regimes, estimate_ou_half_life


def test_regime_classification_covers_all_quadrants():
    qmi = pd.Series([-2.0, -1.0, 1.0, 0.5, -0.5], index=pd.date_range("2020-01-31", periods=5, freq="ME"))
    regimes = classify_regimes(qmi)
    assert regimes.iloc[1] == "Recovery"
    assert regimes.iloc[2] == "Expansion"
    assert regimes.iloc[3] == "Slowdown"
    assert regimes.iloc[4] == "Contraction"


def test_ou_half_life_is_positive_for_mean_reverting_series():
    rng = np.random.default_rng(7)
    values = [0.0]
    for _ in range(500):
        values.append(0.85 * values[-1] + rng.normal(scale=0.2))
    half_life = estimate_ou_half_life(pd.Series(values))
    assert 1.0 < half_life < 20.0

