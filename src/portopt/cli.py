from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from portopt.backtest import run_backtest
from portopt.config import load_config
from portopt.data import download_prices, load_universe
from portopt.macro import build_public_qmi, download_macro
from portopt.regimes import estimate_ou_half_life, gmm_regime_diagnostic
from portopt.reporting import write_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PortOPT research pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--refresh", action="store_true", help="Refresh cached public data")
    parser.add_argument("--output", default="reports")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    universe = load_universe(config["data"]["universe"])
    benchmark_ticker = config["project"]["benchmark"]
    tickers = universe.index.tolist() + [benchmark_ticker]
    prices = download_prices(
        tickers,
        config["project"]["start_date"],
        config["project"]["end_date"] or datetime.now(UTC).date().isoformat(),
        config["data"]["cache_dir"],
        args.refresh,
    )
    if benchmark_ticker not in prices:
        raise RuntimeError(f"Benchmark {benchmark_ticker} was not downloaded")
    benchmark = prices.pop(benchmark_ticker)
    universe = universe.loc[universe.index.intersection(prices.columns)]

    macro = download_macro(config["data"]["cache_dir"], args.refresh)
    qmi_frame = build_public_qmi(macro, int(config["data"]["macro_release_lag_months"]))
    result = run_backtest(prices, benchmark, qmi_frame["qmi"], universe["sector"], config)
    output = Path(config["_root"]) / args.output
    write_reports(result, output)
    gmm_regime_diagnostic(qmi_frame["qmi"]).to_csv(output / "gmm_diagnostic.csv")
    half_life = estimate_ou_half_life(qmi_frame["qmi"])

    print(f"Observations: {len(result.returns):,}")
    print(f"Final value: ${result.metrics['final_value']:,.2f}")
    print(f"CAGR: {result.metrics['cagr']:.2%}")
    print(f"Sharpe: {result.metrics['sharpe']:.2f}")
    print(f"Max drawdown: {result.metrics['max_drawdown']:.2%}")
    print(f"QMI OU half-life: {half_life:.1f} months")
    print(f"Reports: {output}")


if __name__ == "__main__":
    main()
