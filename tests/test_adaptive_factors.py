import pandas as pd

from portopt.factors import walk_forward_factor_weights

CFG = {
    "factor_ic_lookback": 60,
    "factor_ic_halflife": 12,
    "regime_ic_weight": 0.35,
    "minimum_regime_observations": 2,
    "core_factor": "momentum",
    "core_weight": 0.60,
}


def test_walk_forward_weights_ignore_current_and_future_information():
    index = pd.date_range("2020-01-31", periods=6, freq="ME")
    history = pd.DataFrame(
        {
            "momentum": [0.10, 0.20, 0.10, -0.20, -0.50, -0.90],
            "quality": [-0.10, 0.30, 0.20, 0.40, 0.80, 0.90],
        },
        index=index,
    )
    regimes = pd.Series(["Expansion"] * 6, index=index)
    first = walk_forward_factor_weights(history, regimes, index[4], "Expansion", CFG)
    changed = history.copy()
    changed.loc[index[4]:, "quality"] = -100.0
    second = walk_forward_factor_weights(changed, regimes, index[4], "Expansion", CFG)
    pd.testing.assert_series_equal(first, second)


def test_core_factor_keeps_minimum_allocation():
    index = pd.date_range("2020-01-31", periods=6, freq="ME")
    history = pd.DataFrame(
        {"momentum": [-0.2] * 6, "quality": [0.3] * 6}, index=index
    )
    regimes = pd.Series(["Slowdown"] * 6, index=index)
    weights = walk_forward_factor_weights(history, regimes, index[-1], "Slowdown", CFG)
    assert weights["momentum"] >= CFG["core_weight"]
    assert abs(weights.sum() - 1.0) < 1e-12
