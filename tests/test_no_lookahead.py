import numpy as np
import pandas as pd

from portopt.factors import compute_factors


def test_future_prices_do_not_change_past_factor_scores():
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2018-01-01", periods=620)
    returns = rng.normal(0.0002, 0.012, size=(620, 12))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=index)
    benchmark = prices.mean(axis=1)
    cfg = {
        "volatility_lookback": 63,
        "beta_lookback": 126,
        "quality_lookback": 252,
        "momentum_skip": 21,
        "momentum_lookback": 252,
        "value_lookback": 504,
    }
    before = compute_factors(prices, benchmark, cfg)
    changed = prices.copy()
    changed.iloc[-20:] *= 3.0
    after = compute_factors(changed, changed.mean(axis=1), cfg)
    check_date = index[-30]
    for name in before:
        pd.testing.assert_series_equal(before[name].loc[check_date], after[name].loc[check_date])

