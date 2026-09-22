import numpy as np
import pandas as pd

from portopt.optimizer import optimize_weights


def test_optimizer_respects_exposure_and_caps():
    names = [f"S{i}" for i in range(20)]
    score = pd.Series(np.linspace(-2, 2, 20), index=names)
    covariance = pd.DataFrame(np.eye(20) * 0.0002, index=names, columns=names)
    sectors = pd.Series(["Banks"] * 5 + ["Tech"] * 5 + ["Energy"] * 5 + ["Health"] * 5, index=names)
    cfg = {
        "gross_exposure": 1.0,
        "net_exposure_limit": 0.05,
        "max_stock_weight": 0.075,
        "max_sector_gross": 0.30,
        "max_sector_net": 0.10,
        "risk_aversion": 8.0,
        "turnover_penalty": 0.002,
    }
    weights = optimize_weights(score, covariance, sectors, None, cfg, 0.30)
    assert weights.abs().sum() <= 1.00001
    assert abs(weights.sum()) <= 0.05001
    assert weights.abs().max() <= 0.07501
    for sector in sectors.unique():
        selected = weights[sectors == sector]
        assert selected.abs().sum() <= 0.30001
        assert abs(selected.sum()) <= 0.10001

