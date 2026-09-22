import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypdf import PdfReader

from portopt.backtest import BacktestResult
from portopt.metrics import performance_metrics
from portopt.pdf_report import write_portfolio_manager_pdf


def test_portfolio_manager_pdf_contains_required_sections(tmp_path):
    index = pd.bdate_range("2022-01-03", periods=300)
    returns = pd.Series(0.0002 + np.sin(np.arange(300) / 20) * 0.0005, index=index)
    equity = 1_000_000 * (1 + returns).cumprod()
    tickers = ["A", "B", "C", "D", "E", "F"]
    weights = pd.DataFrame(
        [[0.075, 0.075, 0.05, -0.075, -0.075, -0.05]],
        index=[index[0]],
        columns=tickers,
    ).reindex(index).ffill()
    regime_index = pd.date_range(index.min(), index.max(), freq="ME")
    result = BacktestResult(
        equity=equity,
        returns=returns,
        weights=weights,
        turnover=pd.Series(0.0, index=index),
        regimes=pd.Series("Recovery", index=regime_index),
        factor_weights=pd.DataFrame(
            [[0.7, 0.1, 0.1, 0.05, 0.05]],
            columns=["momentum", "value", "quality", "low_risk", "growth"],
            index=[regime_index[-1]],
        ),
        metrics={
            **performance_metrics(returns, equity),
            "average_monthly_turnover": 0.25,
            "average_gross_exposure": 0.40,
            "average_net_exposure": 0.0,
            "final_value": float(equity.iloc[-1]),
        },
    )
    universe = pd.DataFrame(
        {
            "name": [f"Issuer {ticker}" for ticker in tickers],
            "sector": ["Banks", "Banks", "Energy", "Energy", "Technology", "Technology"],
        },
        index=tickers,
    )
    qmi = pd.DataFrame({"qmi": [-0.2, 0.1]}, index=regime_index[-2:])
    config = {
        "project": {"initial_capital": 1_000_000},
        "optimizer": {
            "gross_exposure": 1.0,
            "net_exposure_limit": 0.05,
            "max_stock_weight": 0.075,
            "max_sector_gross": 0.30,
            "max_sector_net": 0.10,
        },
    }
    chart = tmp_path / "chart.png"
    figure, axis = plt.subplots(figsize=(6, 3))
    axis.plot(equity.index, equity)
    figure.savefig(chart, dpi=100)
    plt.close(figure)

    destination = tmp_path / "report.pdf"
    write_portfolio_manager_pdf(result, universe, qmi, 12.0, config, chart, destination)

    reader = PdfReader(destination)
    assert len(reader.pages) >= 4
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "Executive summary" in text
    assert "Current positioning" in text
    assert "Risks, limitations and data provenance" in text
