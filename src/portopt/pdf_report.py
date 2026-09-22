from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from portopt.backtest import BacktestResult

NAVY = colors.HexColor("#12355B")
BLUE = colors.HexColor("#2D6A9F")
PALE_BLUE = colors.HexColor("#EAF2F8")
RED = colors.HexColor("#B23A48")
GREEN = colors.HexColor("#19735A")
INK = colors.HexColor("#1E2933")
MUTED = colors.HexColor("#5D6B78")
LIGHT = colors.HexColor("#F4F6F8")
RULE = colors.HexColor("#D7DEE5")


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=26,
            leading=31,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=16,
            textColor=MUTED,
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "Section",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=NAVY,
            spaceBefore=4,
            spaceAfter=9,
        ),
        "h2": ParagraphStyle(
            "Subsection",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=BLUE,
            spaceBefore=7,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=INK,
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
        ),
        "kpi_label": ParagraphStyle(
            "KpiLabel",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7,
            leading=9,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "kpi_value": ParagraphStyle(
            "KpiValue",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=17,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
    }


def _pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}%}"


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _table(data, widths, header=True, alignments=None, font_size=8):
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 3),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("GRID", (0, 0), (-1, -1), 0.35, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, LIGHT]),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    for column, alignment in enumerate(alignments or []):
        commands.append(("ALIGN", (column, 0), (column, -1), alignment))
    table.setStyle(TableStyle(commands))
    return table


def _header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, height - 14 * mm, width - 18 * mm, height - 14 * mm)
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.setFillColor(NAVY)
    canvas.drawString(18 * mm, height - 10.5 * mm, "PortOPT | Investment Research")
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(width - 18 * mm, height - 10.5 * mm, "Confidential research report")
    canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
    canvas.drawString(18 * mm, 8.5 * mm, "Research use only - not investment advice")
    canvas.drawRightString(width - 18 * mm, 8.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _kpi_grid(result: BacktestResult, styles):
    metrics = result.metrics
    cells = [
        ("ENDING VALUE", _money(metrics["final_value"])),
        ("TOTAL RETURN", _pct(metrics["total_return"])),
        ("CAGR", _pct(metrics["cagr"])),
        ("SHARPE", f"{metrics['sharpe']:.2f}"),
        ("ANNUAL VOLATILITY", _pct(metrics["annual_volatility"])),
        ("MAX DRAWDOWN", _pct(metrics["max_drawdown"])),
        ("AVG GROSS", _pct(metrics["average_gross_exposure"])),
        ("AVG NET", _pct(metrics["average_net_exposure"])),
    ]
    data = []
    for offset in (0, 4):
        data.append([Paragraph(cells[i][0], styles["kpi_label"]) for i in range(offset, offset + 4)])
        data.append([Paragraph(cells[i][1], styles["kpi_value"]) for i in range(offset, offset + 4)])
    table = Table(data, colWidths=[42.5 * mm] * 4, rowHeights=[8 * mm, 12 * mm, 8 * mm, 12 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.6, RULE),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _annual_returns(result: BacktestResult) -> pd.Series:
    return result.returns.groupby(result.returns.index.year).apply(lambda values: (1 + values).prod() - 1)


def _regime_statistics(result: BacktestResult) -> pd.DataFrame:
    regime = result.regimes.reindex(result.returns.index, method="ffill")
    joined = pd.DataFrame({"return": result.returns, "regime": regime}).dropna()
    rows = []
    for name, group in joined.groupby("regime"):
        rows.append(
            {
                "Regime": name,
                "Days": len(group),
                "Annual return": (1 + group["return"]).prod() ** (252 / len(group)) - 1,
                "Annual volatility": group["return"].std() * (252**0.5),
                "Positive days": (group["return"] > 0).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("Regime")


def _position_tables(result: BacktestResult, universe: pd.DataFrame):
    latest = result.weights.iloc[-1].sort_values()
    names = universe["name"].reindex(latest.index).fillna(latest.index.to_series())
    longs = latest[latest > 1e-6].sort_values(ascending=False).head(8)
    shorts = latest[latest < -1e-6].sort_values().head(8)
    long_rows = [[names[ticker], ticker, _pct(weight)] for ticker, weight in longs.items()]
    short_rows = [[names[ticker], ticker, _pct(weight)] for ticker, weight in shorts.items()]

    mapped = pd.DataFrame({"weight": latest, "sector": universe["sector"].reindex(latest.index)})
    sector = mapped.groupby("sector")["weight"].agg(
        net="sum", gross=lambda values: values.abs().sum()
    )
    sector = sector.sort_values("gross", ascending=False)
    sector_rows = [[idx, _pct(row["gross"]), _pct(row["net"])] for idx, row in sector.iterrows()]
    return long_rows, short_rows, sector_rows


def _validate_pdf(path: Path) -> None:
    reader = PdfReader(path)
    if len(reader.pages) < 4:
        raise RuntimeError("Portfolio report did not produce the expected page count")
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    required = [
        "Executive summary",
        "Investment process",
        "Performance and risk",
        "Current positioning",
        "Risks, limitations and data provenance",
    ]
    missing = [heading for heading in required if heading not in text]
    if missing:
        raise RuntimeError(f"Portfolio report is missing sections: {missing}")


def write_portfolio_manager_pdf(
    result: BacktestResult,
    universe: pd.DataFrame,
    qmi: pd.DataFrame,
    half_life: float,
    config: dict,
    performance_chart: str | Path,
    destination: str | Path,
) -> Path:
    """Create and validate a self-contained investment committee PDF."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    start = result.returns.index.min()
    end = result.returns.index.max()
    current_regime = result.regimes.dropna().iloc[-1]
    latest_qmi = qmi["qmi"].dropna().iloc[-1]
    initial = float(config["project"]["initial_capital"])

    doc = SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title="PortOPT Portfolio Manager Report",
        author="Rishabh Raj",
        subject="QMI regime-aware Euro-area equity long/short portfolio research",
    )
    story = []

    story.extend(
        [
            Spacer(1, 8 * mm),
            Paragraph("PortOPT", styles["title"]),
            Paragraph("QMI regime-aware Euro-area equity long/short strategy", styles["subtitle"]),
            Paragraph("Executive summary", styles["h1"]),
            Paragraph(
                f"This report summarizes a monthly rebalanced, market-neutral research portfolio "
                f"initialized with {_money(initial)}. The test spans {start:%d %b %Y} to "
                f"{end:%d %b %Y} and includes transaction costs and short-borrow charges. "
                f"The latest public-data QMI reading is {latest_qmi:.2f}, classified as "
                f"<b>{current_regime}</b>.",
                styles["body"],
            ),
            _kpi_grid(result, styles),
            Spacer(1, 6 * mm),
        ]
    )
    chart = Path(performance_chart)
    if chart.exists():
        story.append(Image(str(chart), width=170 * mm, height=108 * mm))
        story.append(Paragraph("Figure 1. Growth of $1 and drawdown after modeled costs.", styles["small"]))
    story.append(PageBreak())

    constraints = config["optimizer"]
    story.extend(
        [
            Paragraph("Investment process", styles["h1"]),
            Paragraph(
                "The process converts lagged Euro-area macro observations into a four-state QMI "
                "cycle classification: Expansion, Slowdown, Contraction or Recovery. It ranks a "
                "diversified EUR equity universe on momentum, growth, quality, low-risk and "
                "price-based value signals. A 60% momentum core is combined with a 40% satellite "
                "whose factor allocation is estimated from only previously completed forecasts, "
                "including prior observations of the current QMI regime.",
                styles["body"],
            ),
            Paragraph("Decision sequence", styles["h2"]),
            _table(
                [
                    ["Stage", "Control"],
                    ["Macro", "Two-month publication lag; expanding normalization; no future vintages"],
                    ["Signals", "Price data through month-end; 504-day warm-up; cross-sectional z-scores"],
                    ["Factor allocation", "60-month trailing IC; 12-month half-life; QMI-regime conditioning"],
                    ["Portfolio", "Top and bottom 30% of scores; covariance and turnover penalties"],
                    ["Execution", "Target formed at month-end; return accrual begins after next close"],
                    ["Costs", "10 bps per unit of turnover; 50 bps annual short-borrow charge"],
                ],
                [38 * mm, 132 * mm],
            ),
            Paragraph("Mandate constraints", styles["h2"]),
            _table(
                [
                    ["Constraint", "Limit"],
                    ["Gross exposure", f"At most {_pct(float(constraints['gross_exposure']), 0)}"],
                    ["Absolute net exposure", f"At most {_pct(float(constraints['net_exposure_limit']), 0)}"],
                    ["Individual position", f"At most {_pct(float(constraints['max_stock_weight']), 1)}"],
                    ["Sector gross exposure", f"At most {_pct(float(constraints['max_sector_gross']), 0)}"],
                    ["Absolute sector net", f"At most {_pct(float(constraints['max_sector_net']), 0)}"],
                ],
                [85 * mm, 85 * mm],
                alignments=["LEFT", "RIGHT"],
            ),
            Paragraph("Macro diagnostics", styles["h2"]),
            Paragraph(
                f"The estimated QMI Ornstein-Uhlenbeck half-life is {half_life:.1f} months. This is "
                "reported as a persistence diagnostic, not used to choose the monthly rebalance "
                "frequency. Gaussian-mixture classifications are likewise diagnostic only.",
                styles["body"],
            ),
            PageBreak(),
        ]
    )

    annual = _annual_returns(result)
    annual_rows = [[str(year), _pct(value)] for year, value in annual.items()]
    regime_stats = _regime_statistics(result)
    regime_rows = [
        [
            row["Regime"],
            f"{int(row['Days']):,}",
            _pct(row["Annual return"]),
            _pct(row["Annual volatility"]),
            _pct(row["Positive days"]),
        ]
        for _, row in regime_stats.iterrows()
    ]
    story.extend(
        [
            Paragraph("Performance and risk", styles["h1"]),
            Paragraph(
                "Returns below are net of modeled turnover and borrow costs. The strategy is "
                "market-neutral by mandate, so it should be evaluated primarily on consistency, "
                "drawdown and risk-adjusted return rather than equity-market beta.",
                styles["body"],
            ),
            KeepTogether(
                [
                    Paragraph("Calendar-year returns", styles["h2"]),
                    _table(
                        [["Year", "Net return"], *annual_rows],
                        [85 * mm, 85 * mm],
                        alignments=["LEFT", "RIGHT"],
                    ),
                ]
            ),
            Paragraph("Performance by QMI regime", styles["h2"]),
            _table(
                [["Regime", "Days", "Annual return", "Annual volatility", "Positive days"], *regime_rows],
                [36 * mm, 22 * mm, 37 * mm, 42 * mm, 33 * mm],
                alignments=["LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"],
                font_size=7.5,
            ),
            Paragraph("Interpretation", styles["h2"]),
            Paragraph(
                "The full-history result is positive but modest, and the maximum drawdown remains "
                "material. Recent performance is stronger than the early sample. This divergence "
                "should be treated as a prompt for continued live validation, not evidence that the "
                "signal has permanently improved.",
                styles["body"],
            ),
            PageBreak(),
        ]
    )

    long_rows, short_rows, sector_rows = _position_tables(result, universe)
    factors = result.factor_weights.iloc[-1].sort_values(ascending=False)
    factor_rows = [[name.replace("_", " ").title(), _pct(weight)] for name, weight in factors.items()]
    story.extend(
        [
            Paragraph("Current positioning", styles["h1"]),
            Paragraph(
                f"Snapshot as of {end:%d %b %Y}. Weights reflect the latest modeled allocation "
                "and should not be interpreted as a live recommendation.",
                styles["body"],
            ),
            Paragraph("Current factor allocation", styles["h2"]),
            _table(
                [["Factor", "Weight"], *factor_rows],
                [85 * mm, 85 * mm],
                alignments=["LEFT", "RIGHT"],
            ),
            Paragraph("Sector exposure", styles["h2"]),
            _table(
                [["Sector", "Gross", "Net"], *sector_rows],
                [90 * mm, 40 * mm, 40 * mm],
                alignments=["LEFT", "RIGHT", "RIGHT"],
                font_size=7.5,
            ),
            PageBreak(),
            Paragraph("Position detail", styles["h1"]),
            Paragraph("Largest long positions", styles["h2"]),
            _table(
                [["Issuer", "Ticker", "Weight"], *long_rows],
                [92 * mm, 38 * mm, 40 * mm],
                alignments=["LEFT", "LEFT", "RIGHT"],
            ),
            Paragraph("Largest short positions", styles["h2"]),
            _table(
                [["Issuer", "Ticker", "Weight"], *short_rows],
                [92 * mm, 38 * mm, 40 * mm],
                alignments=["LEFT", "LEFT", "RIGHT"],
            ),
            PageBreak(),
        ]
    )

    limitations = [
        "The static security universe introduces survivorship bias and does not include delisting returns.",
        "Price-derived quality and value signals are proxies, not point-in-time accounting fundamentals.",
        "A conservative release lag is applied, but macro vintage revisions are not fully reconstructed.",
        "Borrow availability, dividends on shorts, withholding taxes, market impact and capacity are simplified.",
        "Historical positive returns do not establish that future returns will be positive.",
    ]
    sources = [
        "ECB Data Portal API: deposit facility rate.",
        "FRED distribution of OECD and Euro-area business confidence, unemployment and inflation series.",
        "Yahoo Finance adjusted price histories accessed through the open-source yfinance client.",
    ]
    story.extend(
        [
            Paragraph("Risks, limitations and data provenance", styles["h1"]),
            Paragraph("Material limitations", styles["h2"]),
            *[Paragraph(f"- {item}", styles["body"]) for item in limitations],
            Paragraph("Data provenance", styles["h2"]),
            *[Paragraph(f"- {item}", styles["body"]) for item in sources],
            Paragraph("Governance and reproducibility", styles["h2"]),
            Paragraph(
                "The pipeline caches downloaded data, exports daily returns, holdings, turnover, "
                "regimes and factor weights, and validates constraint behavior through automated "
                "tests. The PDF is generated from those same in-memory results at the end of each "
                "successful run. Configuration values are stored in configs/default.yaml.",
                styles["body"],
            ),
            Spacer(1, 8 * mm),
            Paragraph("Important notice", styles["h2"]),
            Paragraph(
                "This document is a research artifact for discussion with a portfolio manager. It "
                "does not constitute investment advice, an offer, or a recommendation to transact. "
                "Independent validation, licensed production data and pre-trade risk controls are "
                "required before any live deployment.",
                styles["body"],
            ),
            Spacer(1, 12 * mm),
            Paragraph(
                f"Report generated {datetime.now().astimezone():%d %b %Y, %H:%M %Z}",
                styles["small"],
            ),
        ]
    )

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    _validate_pdf(destination)
    return destination
