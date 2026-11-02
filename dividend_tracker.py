#!/usr/bin/env python3
"""
Automated Dividend Growth Tracker & Visualizer
===============================================

Pulls live prices and dividend histories from Yahoo Finance (via yfinance),
computes income/safety metrics for a dividend portfolio, and renders three
interactive Plotly charts plus a combined HTML dashboard:

    1. 12-Month Dividend Ladder      - estimated cash flow per month, by ticker
    2. Weight vs. Income Contribution - flags income concentration risk
    3. 10-Year Compound Projection    - cash payout vs. full DRIP reinvestment

Usage
-----
    python dividend_tracker.py                      # run with the sample portfolio
    python dividend_tracker.py --portfolio my.csv   # CSV with Ticker,Shares,AvgCost
    python dividend_tracker.py --show               # also open charts in a browser
    python dividend_tracker.py --help               # all options

Disclaimer: this is an analytical tool, not investment advice. Yahoo Finance
data can be delayed, revised, or missing; verify figures before acting on them.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
import warnings
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SAMPLE_PORTFOLIO = pd.DataFrame(
    {
        "Ticker": ["SCHD", "O", "AAPL", "MSFT", "XOM"],
        "Shares": [150, 120, 40, 25, 60],
        "AvgCost": [25.50, 54.00, 150.00, 310.00, 95.00],
    }
)

# Payout-ratio thresholds above which a dividend is flagged as risky.
CORPORATE_PAYOUT_LIMIT = 0.75
REIT_PAYOUT_LIMIT = 0.90

# A single position producing more than this share of total income is flagged.
INCOME_CONCENTRATION_LIMIT = 0.25

# Chart styling: a validated colorblind-safe categorical order (assigned in
# fixed order, never cycled) plus recessive ink/grid tones.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                 "#008300", "#6250d6", "#e34948"]
OTHER_COLOR = "#8a8984"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"
FONT_FAMILY = "Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"

FUND_QUOTE_TYPES = {"ETF", "MUTUALFUND", "CLOSEDENDFUND"}

log = logging.getLogger("dividend_tracker")


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

@dataclass
class TickerData:
    """Everything fetched (or derived) for one ticker."""

    ticker: str
    name: str = ""
    quote_type: str = "EQUITY"
    sector: str = ""
    industry: str = ""
    price: float = float("nan")
    forward_dividend: float = 0.0          # annual dividend per share
    trailing_dividend: float = 0.0         # sum of last 365 days of dividends
    payout_ratio: float = float("nan")     # fraction, e.g. 0.45 = 45%
    trailing_eps: float = float("nan")
    dividend_cagr_5y: float = float("nan")
    frequency: int = 0                     # payments per year
    pay_months: list[int] = field(default_factory=list)
    dividends: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    notes: list[str] = field(default_factory=list)

    @property
    def is_fund(self) -> bool:
        return self.quote_type.upper() in FUND_QUOTE_TYPES

    @property
    def is_reit(self) -> bool:
        return "REIT" in self.industry.upper() or self.sector == "Real Estate"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _safe_float(value) -> float:
    """Convert a yfinance field to float, mapping None/garbage/inf to NaN."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _with_retries(func, description: str, attempts: int = 3, backoff: float = 2.0):
    """Call func(), retrying with exponential backoff on any exception."""
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # yfinance raises a wide variety of errors
            if attempt == attempts:
                log.warning("%s failed after %d attempts: %s", description, attempts, exc)
                return None
            wait = backoff ** attempt
            log.info("%s failed (%s); retrying in %.0fs", description, exc, wait)
            time.sleep(wait)
    return None


def _infer_payment_schedule(dividends: pd.Series, pay_lag_days: int) -> tuple[int, list[int]]:
    """
    Infer payment frequency and calendar months from dividend history.

    Frequency comes from the median gap between the last ~3 years of ex-dates
    (robust to one-off special dividends). Months are laid out evenly from the
    most recent payment, shifted by `pay_lag_days` to approximate the pay date
    (yfinance only exposes ex-dividend dates).
    """
    if dividends.empty:
        return 0, []

    cutoff = dividends.index.max() - pd.DateOffset(years=3)
    recent = dividends[dividends.index >= cutoff]

    if len(recent) < 2:
        frequency = 1
    else:
        median_gap = recent.index.to_series().diff().dt.days.dropna().median()
        if median_gap <= 45:
            frequency = 12
        elif median_gap <= 120:
            frequency = 4
        elif median_gap <= 240:
            frequency = 2
        else:
            frequency = 1

    last_pay = recent.index.max() + pd.Timedelta(days=pay_lag_days)
    step = 12 // frequency
    months = sorted({(last_pay.month - 1 + k * step) % 12 + 1 for k in range(frequency)})
    return frequency, months


def _dividend_cagr(dividends: pd.Series, years: int = 5) -> float:
    """CAGR of annual dividends over the last `years` complete calendar years."""
    if dividends.empty:
        return float("nan")
    annual = dividends.groupby(dividends.index.year).sum()
    current_year = pd.Timestamp.now().year
    complete = annual[annual.index < current_year]
    if len(complete) < years + 1:
        return float("nan")
    end_year = complete.index.max()
    start_year = end_year - years
    if start_year not in complete.index or complete[start_year] <= 0:
        return float("nan")
    return (complete[end_year] / complete[start_year]) ** (1 / years) - 1


def _fmt_money(x: float) -> str:
    return "—" if pd.isna(x) else f"${x:,.2f}"


def _fmt_pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:.2%}"


# --------------------------------------------------------------------------- #
# Main class
# --------------------------------------------------------------------------- #

class PortfolioTracker:
    """Fetches market data, computes dividend metrics and builds charts."""

    def __init__(
        self,
        portfolio: pd.DataFrame,
        price_growth: float = 0.05,
        dividend_growth: float = 0.06,
        projection_years: int = 10,
        pay_lag_days: int = 0,
        concentration_limit: float = INCOME_CONCENTRATION_LIMIT,
    ):
        self.portfolio = self._validate_portfolio(portfolio)
        self.price_growth = price_growth
        self.dividend_growth = dividend_growth
        self.projection_years = projection_years
        self.pay_lag_days = pay_lag_days
        self.concentration_limit = concentration_limit

        self.data: dict[str, TickerData] = {}
        self.skipped: list[str] = []
        self.summary: pd.DataFrame | None = None
        self.projection: pd.DataFrame | None = None
        self.colors: dict[str, str] = {}

    # ----------------------------- input ---------------------------------- #

    @staticmethod
    def _validate_portfolio(portfolio: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names, types and duplicates in the input portfolio."""
        df = portfolio.copy()
        df.columns = [c.strip() for c in df.columns]
        rename = {c: "AvgCost" for c in df.columns
                  if c.lower().replace(" ", "").replace("_", "") in {"avgcost", "averagepurchaseprice", "avgprice"}}
        rename.update({c: "Shares" for c in df.columns if c.lower() in {"shares", "sharesowned", "shares owned"}})
        rename.update({c: "Ticker" for c in df.columns if c.lower() in {"ticker", "symbol"}})
        df = df.rename(columns=rename)

        missing = {"Ticker", "Shares", "AvgCost"} - set(df.columns)
        if missing:
            raise ValueError(f"Portfolio is missing required columns: {sorted(missing)}")

        df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
        df["Shares"] = pd.to_numeric(df["Shares"], errors="coerce")
        df["AvgCost"] = pd.to_numeric(df["AvgCost"], errors="coerce")

        bad = df[df["Shares"].isna() | (df["Shares"] <= 0) | df["AvgCost"].isna() | (df["AvgCost"] < 0)]
        if not bad.empty:
            log.warning("Dropping rows with invalid Shares/AvgCost: %s", bad["Ticker"].tolist())
            df = df.drop(bad.index)

        # Merge duplicate lots into one position with a share-weighted average cost.
        df["Cost"] = df["Shares"] * df["AvgCost"]
        df = df.groupby("Ticker", as_index=False, sort=False).agg(Shares=("Shares", "sum"), Cost=("Cost", "sum"))
        df["AvgCost"] = df["Cost"] / df["Shares"]
        df = df.drop(columns="Cost")

        if df.empty:
            raise ValueError("Portfolio has no valid positions.")
        return df

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs) -> "PortfolioTracker":
        """Build a tracker from a CSV with columns Ticker, Shares, AvgCost."""
        return cls(pd.read_csv(path), **kwargs)

    # ----------------------------- fetch ---------------------------------- #

    def fetch_data(self) -> None:
        """Download price, dividend and fundamentals data for every ticker."""
        for ticker in self.portfolio["Ticker"]:
            log.info("Fetching %s ...", ticker)
            td = self._fetch_ticker(ticker)
            if pd.isna(td.price) or td.price <= 0:
                log.warning("No usable price for %s; excluding it from the analysis.", ticker)
                self.skipped.append(ticker)
                continue
            self.data[ticker] = td

        if not self.data:
            raise RuntimeError("Could not fetch market data for any ticker. Check your connection or symbols.")

        self.portfolio = self.portfolio[self.portfolio["Ticker"].isin(self.data)].reset_index(drop=True)
        # Colors follow the entity: assigned once, in portfolio order.
        for i, ticker in enumerate(self.portfolio["Ticker"]):
            self.colors[ticker] = SERIES_COLORS[i] if i < len(SERIES_COLORS) else OTHER_COLOR

    def _fetch_ticker(self, ticker: str) -> TickerData:
        td = TickerData(ticker=ticker)
        yt = yf.Ticker(ticker)

        info = _with_retries(lambda: yt.info, f"{ticker} info") or {}
        td.name = info.get("shortName") or info.get("longName") or ticker
        td.quote_type = (info.get("quoteType") or "EQUITY").upper()
        td.sector = info.get("sector") or ""
        td.industry = info.get("industry") or ""
        td.trailing_eps = _safe_float(info.get("trailingEps"))

        td.price = self._resolve_price(yt, info, td)

        dividends = _with_retries(lambda: yt.dividends, f"{ticker} dividends")
        if isinstance(dividends, pd.Series) and not dividends.empty:
            dividends = dividends[dividends > 0].sort_index()
            if dividends.index.tz is not None:
                dividends.index = dividends.index.tz_localize(None)
            td.dividends = dividends
        else:
            td.notes.append("no dividend history")

        now = pd.Timestamp.now()
        if not td.dividends.empty:
            td.trailing_dividend = float(td.dividends[td.dividends.index >= now - pd.Timedelta(days=365)].sum())
            td.frequency, td.pay_months = _infer_payment_schedule(td.dividends, self.pay_lag_days)
            td.dividend_cagr_5y = _dividend_cagr(td.dividends)

        td.forward_dividend = self._resolve_forward_dividend(info, td, now)
        if td.forward_dividend > 0 and not td.pay_months:
            # Paying a dividend but no usable history: assume a standard quarterly calendar.
            td.frequency, td.pay_months = 4, [3, 6, 9, 12]
            td.notes.append("schedule assumed quarterly")
        if td.forward_dividend == 0:
            td.frequency, td.pay_months = 0, []

        td.payout_ratio = self._resolve_payout_ratio(info, td)
        return td

    @staticmethod
    def _resolve_price(yt: yf.Ticker, info: dict, td: TickerData) -> float:
        """Latest price: fast_info -> info fields -> last close of recent history."""
        fast = _with_retries(lambda: yt.fast_info.last_price, f"{td.ticker} fast_info", attempts=2)
        price = _safe_float(fast)
        if price > 0:
            return price

        for key in ("currentPrice", "regularMarketPrice", "navPrice", "previousClose"):
            price = _safe_float(info.get(key))
            if price > 0:
                td.notes.append(f"price from info.{key}")
                return price

        hist = _with_retries(lambda: yt.history(period="5d"), f"{td.ticker} history")
        if isinstance(hist, pd.DataFrame) and not hist.empty and "Close" in hist:
            td.notes.append("price from last close")
            return _safe_float(hist["Close"].dropna().iloc[-1])
        return float("nan")

    @staticmethod
    def _resolve_forward_dividend(info: dict, td: TickerData, now: pd.Timestamp) -> float:
        """
        Annual forward dividend per share.

        Priority: Yahoo's forward `dividendRate` (declared rate x frequency) ->
        trailing-12-month sum from history -> Yahoo's `trailingAnnualDividendRate`.
        A history whose last payment is >400 days old is treated as suspended.
        """
        last_paid = td.dividends.index.max() if not td.dividends.empty else None
        suspended = last_paid is not None and (now - last_paid).days > 400

        rate = _safe_float(info.get("dividendRate"))
        if rate > 0:
            return rate
        if suspended:
            td.notes.append("dividend appears suspended")
            return 0.0
        if td.trailing_dividend > 0:
            td.notes.append("forward div = TTM history")
            return td.trailing_dividend
        rate = _safe_float(info.get("trailingAnnualDividendRate"))
        if rate > 0:
            td.notes.append("forward div = info TTM rate")
            return rate
        return 0.0

    @staticmethod
    def _resolve_payout_ratio(info: dict, td: TickerData) -> float:
        """Payout ratio from Yahoo, falling back to TTM dividends / trailing EPS."""
        if td.is_fund or td.forward_dividend == 0:
            return float("nan")
        ratio = _safe_float(info.get("payoutRatio"))
        if ratio > 0:
            return ratio
        dividends = td.trailing_dividend or td.forward_dividend
        if td.trailing_eps > 0:
            td.notes.append("payout = TTM div / EPS")
            return dividends / td.trailing_eps
        return float("nan")

    # ----------------------------- metrics -------------------------------- #

    def _safety_flag(self, td: TickerData) -> tuple[str, float]:
        """Return (flag, threshold) for a position's dividend safety."""
        if td.forward_dividend == 0:
            return "NO DIVIDEND", float("nan")
        if td.is_fund:
            return "N/A (FUND)", float("nan")
        threshold = REIT_PAYOUT_LIMIT if td.is_reit else CORPORATE_PAYOUT_LIMIT
        if pd.isna(td.payout_ratio):
            if td.trailing_eps <= 0:
                return "RISKY (EPS ≤ 0)", threshold
            return "UNKNOWN", threshold
        return ("RISKY" if td.payout_ratio > threshold else "SAFE"), threshold

    def compute_metrics(self) -> pd.DataFrame:
        """Build the per-position summary table with a portfolio TOTAL row."""
        if not self.data:
            self.fetch_data()

        rows = []
        for _, pos in self.portfolio.iterrows():
            td = self.data[pos["Ticker"]]
            flag, threshold = self._safety_flag(td)
            market_value = pos["Shares"] * td.price
            cost_basis = pos["Shares"] * pos["AvgCost"]
            income = pos["Shares"] * td.forward_dividend
            rows.append({
                "Ticker": td.ticker,
                "Name": td.name,
                "Type": "REIT" if td.is_reit else ("Fund" if td.is_fund else "Stock"),
                "Shares": pos["Shares"],
                "Avg Cost": pos["AvgCost"],
                "Price": td.price,
                "Cost Basis": cost_basis,
                "Market Value": market_value,
                "Unrealized G/L": market_value - cost_basis,
                "G/L %": (market_value / cost_basis - 1) if cost_basis > 0 else np.nan,
                "Fwd Div/Share": td.forward_dividend,
                "Annual Income": income,
                "Current Yield": td.forward_dividend / td.price,
                "Yield on Cost": td.forward_dividend / pos["AvgCost"] if pos["AvgCost"] > 0 else np.nan,
                "Payout Ratio": td.payout_ratio,
                "Payout Limit": threshold,
                "Safety Flag": flag,
                "Div CAGR 5Y": td.dividend_cagr_5y,
                "Payments/Yr": td.frequency,
                "Pay Months": "Monthly" if td.frequency == 12 else
                              (", ".join(pd.Timestamp(2000, m, 1).strftime("%b") for m in td.pay_months) or "—"),
                "Notes": "; ".join(td.notes),
            })

        df = pd.DataFrame(rows)
        total_value = df["Market Value"].sum()
        total_cost = df["Cost Basis"].sum()
        total_income = df["Annual Income"].sum()
        df["Weight %"] = df["Market Value"] / total_value
        df["Income %"] = df["Annual Income"] / total_income if total_income > 0 else 0.0

        total = {
            "Ticker": "TOTAL", "Name": "Portfolio", "Type": "",
            "Cost Basis": total_cost,
            "Market Value": total_value,
            "Unrealized G/L": total_value - total_cost,
            "G/L %": total_value / total_cost - 1 if total_cost > 0 else np.nan,
            "Annual Income": total_income,
            "Current Yield": total_income / total_value,
            "Yield on Cost": total_income / total_cost if total_cost > 0 else np.nan,
            "Weight %": 1.0,
            "Income %": 1.0 if total_income > 0 else 0.0,
            "Safety Flag": f"{(df['Safety Flag'].str.startswith('RISKY')).sum()} risky",
            "Pay Months": "", "Notes": "",
        }
        self.summary = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
        return self.summary

    def positions(self) -> pd.DataFrame:
        """Summary without the TOTAL row."""
        if self.summary is None:
            self.compute_metrics()
        return self.summary[self.summary["Ticker"] != "TOTAL"].reset_index(drop=True)

    def print_summary(self) -> None:
        """Pretty-print the key metrics to the console."""
        s = self.summary if self.summary is not None else self.compute_metrics()
        view = s[["Ticker", "Shares", "Price", "Cost Basis", "Market Value", "G/L %", "Fwd Div/Share",
                  "Annual Income", "Current Yield", "Yield on Cost", "Payout Ratio", "Safety Flag",
                  "Weight %", "Income %", "Pay Months"]].copy()
        money = ["Price", "Cost Basis", "Market Value", "Fwd Div/Share", "Annual Income"]
        pct = ["G/L %", "Current Yield", "Yield on Cost", "Payout Ratio", "Weight %", "Income %"]
        for col in money:
            view[col] = view[col].map(_fmt_money)
        for col in pct:
            view[col] = view[col].map(_fmt_pct)
        view["Shares"] = view["Shares"].map(lambda x: "" if pd.isna(x) else f"{x:,.4g}")

        with pd.option_context("display.max_columns", None, "display.width", 250):
            print("\n" + "=" * 100)
            print("DIVIDEND PORTFOLIO SUMMARY  ·  data as of", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
            print("=" * 100)
            print(view.to_string(index=False))

        for _, row in self.positions().iterrows():
            if row["Safety Flag"].startswith("RISKY"):
                limit = "" if pd.isna(row["Payout Limit"]) else f" (limit {row['Payout Limit']:.0%})"
                print(f"  ⚠ {row['Ticker']}: payout ratio {_fmt_pct(row['Payout Ratio'])}{limit} — {row['Safety Flag']}")
            if row["Income %"] > self.concentration_limit:
                print(f"  ⚠ {row['Ticker']}: produces {row['Income %']:.1%} of portfolio income "
                      f"(concentration limit {self.concentration_limit:.0%})")
            if row["Notes"]:
                print(f"  · {row['Ticker']}: {row['Notes']}")
        if self.skipped:
            print(f"  ⚠ Skipped (no market data): {', '.join(self.skipped)}")

    # ----------------------------- schedule ------------------------------- #

    def monthly_schedule(self) -> pd.DataFrame:
        """
        Estimated dividend cash flow for the next 12 months (rows = months,
        columns = tickers). Each payment = shares x forward dividend / frequency.
        """
        pos = self.positions()
        start = pd.Timestamp.now().normalize().replace(day=1)
        months = pd.date_range(start, periods=12, freq="MS")
        schedule = pd.DataFrame(0.0, index=months, columns=pos["Ticker"].tolist())
        for _, row in pos.iterrows():
            td = self.data[row["Ticker"]]
            if td.frequency == 0:
                continue
            payment = row["Shares"] * td.forward_dividend / td.frequency
            for month in months:
                if month.month in td.pay_months:
                    schedule.loc[month, row["Ticker"]] = payment
        return schedule

    # ----------------------------- projection ----------------------------- #

    def project(self) -> pd.DataFrame:
        """
        Month-by-month projection of two strategies over `projection_years`.

        Series A (cash): share count is fixed; dividends accumulate as
            uninvested cash. Value = shares x price + cumulative cash.
        Series B (DRIP): each payment buys fractional shares at that month's
            price, so compounding happens on every pay date (monthly for O,
            quarterly for most others).

        Price grows at `price_growth`/yr (compounded monthly). Dividend per
        share steps up by `dividend_growth` once every 12 months.
        """
        pos = self.positions()
        n_months = self.projection_years * 12
        start = pd.Timestamp.now().normalize().replace(day=1)
        dates = pd.date_range(start, periods=n_months + 1, freq="MS")
        monthly_price_factor = (1 + self.price_growth) ** (1 / 12)

        shares0 = pos["Shares"].to_numpy(dtype=float)
        price0 = pos["Price"].to_numpy(dtype=float)
        dps0 = pos["Fwd Div/Share"].to_numpy(dtype=float)
        freq = np.array([self.data[t].frequency for t in pos["Ticker"]], dtype=float)
        pays = [set(self.data[t].pay_months) for t in pos["Ticker"]]

        shares_drip = shares0.copy()
        cash_a = 0.0
        records = [{
            "Date": dates[0], "Value_A": float(shares0 @ price0), "Value_B": float(shares0 @ price0),
            "Cash_A": 0.0, "Income_A": 0.0, "Income_B": 0.0, "Shares_B_Factor": 1.0,
        }]

        for t in range(1, n_months + 1):
            date = dates[t]
            price_t = price0 * monthly_price_factor ** t
            dps_t = dps0 * (1 + self.dividend_growth) ** ((t - 1) // 12)
            paying = np.array([date.month in p for p in pays]) & (freq > 0)
            per_share = np.where(paying, dps_t / np.where(freq > 0, freq, 1), 0.0)

            income_a = float(shares0 @ per_share)
            income_b = float(shares_drip @ per_share)
            cash_a += income_a
            shares_drip = shares_drip + shares_drip * per_share / price_t  # DRIP at month's price

            records.append({
                "Date": date,
                "Value_A": float(shares0 @ price_t) + cash_a,
                "Value_B": float(shares_drip @ price_t),
                "Cash_A": cash_a,
                "Income_A": income_a,
                "Income_B": income_b,
                # Growth in holdings from reinvestment alone (value-weighted).
                "Shares_B_Factor": float((shares_drip @ price_t) / (shares0 @ price_t)),
            })

        self.projection = pd.DataFrame(records).set_index("Date")
        return self.projection

    # ----------------------------- charts --------------------------------- #

    @staticmethod
    def _base_layout(fig: go.Figure, title: str, subtitle: str, height: int = 520) -> go.Figure:
        # Escape "$" so Plotly never mistakes "$a ... $b" for a LaTeX expression.
        subtitle = subtitle.replace("$", "&#36;")
        fig.update_layout(
            title=dict(text=f"<b>{title}</b><br><span style='font-size:13px;color:{TEXT_SECONDARY}'>{subtitle}</span>",
                       x=0.01, xanchor="left", y=0.96),
            template="plotly_white",
            paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
            font=dict(family=FONT_FAMILY, size=13, color=TEXT_PRIMARY),
            height=height, margin=dict(l=70, r=40, t=110, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                        font=dict(color=TEXT_SECONDARY)),
            hoverlabel=dict(bgcolor="white", font=dict(family=FONT_FAMILY, color=TEXT_PRIMARY)),
            barcornerradius=4,
        )
        fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=TEXT_SECONDARY))
        fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=TEXT_SECONDARY))
        return fig

    def chart_dividend_ladder(self) -> go.Figure:
        """Chart 1: stacked monthly bar chart of estimated dividend income."""
        schedule = self.monthly_schedule()
        labels = schedule.index.strftime("%b %Y")
        fig = go.Figure()
        for ticker in schedule.columns:
            if schedule[ticker].sum() == 0:
                continue
            fig.add_bar(
                x=labels, y=schedule[ticker], name=ticker,
                marker=dict(color=self.colors[ticker], line=dict(color=SURFACE, width=1.5)),
                hovertemplate=f"<b>{ticker}</b><br>%{{x}}: $%{{y:,.2f}}<extra></extra>",
            )

        totals = schedule.sum(axis=1)
        avg = totals.mean()
        fig.add_scatter(
            x=labels, y=totals, mode="text", text=[f"${v:,.0f}" if v > 0 else "" for v in totals],
            textposition="top center", textfont=dict(color=TEXT_SECONDARY, size=12),
            showlegend=False, hovertemplate="<b>Total</b> %{x}: $%{y:,.2f}<extra></extra>",
        )
        fig.add_hline(y=avg, line=dict(color=TEXT_SECONDARY, width=1, dash="dot"),
                      annotation_text=f"Avg ${avg:,.0f}/mo", annotation_position="top left",
                      annotation_font=dict(color=TEXT_SECONDARY, size=12))
        fig.update_layout(barmode="stack", bargap=0.25, legend_traceorder="normal")
        fig.update_yaxes(title_text="Estimated dividend income", tickprefix="$", tickformat=",.0f",
                         range=[0, totals.max() * 1.18 if totals.max() > 0 else 1])
        lag_note = f"ex-date + {self.pay_lag_days} days" if self.pay_lag_days else "by ex-dividend month"
        return self._base_layout(
            fig, "12-Month Dividend Ladder",
            f"Estimated cash flow ({lag_note}) · ${totals.sum():,.0f} expected over the next 12 months",
        )

    def chart_weight_vs_income(self) -> go.Figure:
        """Chart 2: portfolio weight vs. income contribution, side by side."""
        pos = self.positions().sort_values("Income %", ascending=True)
        flagged = pos["Income %"] > self.concentration_limit
        y_labels = [f"⚠ {t}" if f else t for t, f in zip(pos["Ticker"], flagged)]

        fig = go.Figure()
        fig.add_bar(
            y=y_labels, x=pos["Weight %"], orientation="h", name="Portfolio weight",
            marker=dict(color=SERIES_COLORS[0], line=dict(color=SURFACE, width=2)),
            customdata=pos["Market Value"],
            hovertemplate="<b>%{y}</b><br>Weight: %{x:.1%}<br>Value: $%{customdata:,.0f}<extra></extra>",
        )
        fig.add_bar(
            y=y_labels, x=pos["Income %"], orientation="h", name="Income contribution",
            marker=dict(color=SERIES_COLORS[1], line=dict(color=SURFACE, width=2)),
            customdata=pos["Annual Income"],
            hovertemplate="<b>%{y}</b><br>Income share: %{x:.1%}<br>Income: $%{customdata:,.0f}/yr<extra></extra>",
        )
        for label, w, inc in zip(y_labels, pos["Weight %"], pos["Income %"]):
            fig.add_annotation(x=max(w, inc), y=label, text=f"{w:.0%} → {inc:.0%}", showarrow=False,
                               xanchor="left", xshift=8, font=dict(size=12, color=TEXT_SECONDARY))
        fig.add_vline(x=self.concentration_limit, line=dict(color=TEXT_SECONDARY, width=1, dash="dash"),
                      annotation_text=f"Income concentration limit {self.concentration_limit:.0%}",
                      annotation_position="top", annotation_font=dict(color=TEXT_SECONDARY, size=12))

        max_x = max(pos["Weight %"].max(), pos["Income %"].max(), self.concentration_limit)
        fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
        fig.update_xaxes(tickformat=".0%", range=[0, max_x * 1.25], showgrid=True, gridcolor=GRID)
        fig.update_yaxes(showgrid=False)

        if flagged.any():
            names = ", ".join(pos.loc[flagged, "Ticker"])
            subtitle = f"⚠ {names} exceed{'s' if flagged.sum() == 1 else ''} the {self.concentration_limit:.0%} income limit"
        else:
            subtitle = f"No position exceeds the {self.concentration_limit:.0%} income concentration limit"
        return self._base_layout(fig, "Portfolio Weight vs. Dividend Income Contribution", subtitle,
                                 height=max(420, 90 * len(pos) + 200))

    def chart_projection(self) -> go.Figure:
        """Chart 3: 10-year value curves plus annual income, cash vs. DRIP."""
        proj = self.projection if self.projection is not None else self.project()
        color_a, color_b = SERIES_COLORS[0], SERIES_COLORS[1]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=False, row_heights=[0.66, 0.34],
                            vertical_spacing=0.12,
                            subplot_titles=("Total portfolio value", "Annual dividend income"))
        fig.add_scatter(
            x=proj.index, y=proj["Value_A"], name="A · Dividends taken as cash",
            line=dict(color=color_a, width=2), row=1, col=1,
            customdata=proj["Cash_A"],
            hovertemplate="A (cash): $%{y:,.0f}<br>incl. cash collected $%{customdata:,.0f}<extra></extra>",
        )
        fig.add_scatter(
            x=proj.index, y=proj["Value_B"], name="B · Full DRIP reinvestment",
            line=dict(color=color_b, width=2), row=1, col=1,
            customdata=proj["Shares_B_Factor"],
            hovertemplate="B (DRIP): $%{y:,.0f}<br>share count ×%{customdata:.2f}<extra></extra>",
        )

        end = proj.iloc[-1]
        for series, color in (("Value_A", color_a), ("Value_B", color_b)):
            fig.add_annotation(x=proj.index[-1], y=end[series], text=f"<b>${end[series]:,.0f}</b>",
                               showarrow=False, xanchor="left", xshift=6, font=dict(color=TEXT_PRIMARY),
                               row=1, col=1)
        fig.add_annotation(x=proj.index[0], y=proj["Value_A"].iloc[0], text=f"Today ${proj['Value_A'].iloc[0]:,.0f}",
                           showarrow=False, xanchor="left", yshift=14, font=dict(color=TEXT_SECONDARY, size=12),
                           row=1, col=1)

        yearly = proj.iloc[1:].groupby((np.arange(len(proj) - 1) // 12) + 1)[["Income_A", "Income_B"]].sum()
        year_labels = [f"Yr {y}" for y in yearly.index]
        fig.add_bar(x=year_labels, y=yearly["Income_A"], name="Income A (cash)", showlegend=False,
                    marker=dict(color=color_a, line=dict(color=SURFACE, width=2)), row=2, col=1,
                    hovertemplate="%{x} · A (cash): $%{y:,.0f}<extra></extra>")
        fig.add_bar(x=year_labels, y=yearly["Income_B"], name="Income B (DRIP)", showlegend=False,
                    marker=dict(color=color_b, line=dict(color=SURFACE, width=2)), row=2, col=1,
                    hovertemplate="%{x} · B (DRIP): $%{y:,.0f}<extra></extra>")

        fig.update_layout(barmode="group", bargap=0.25, hovermode="x unified")
        fig.update_yaxes(tickprefix="$", tickformat=",.0f", gridcolor=GRID)
        fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1, spikecolor=TEXT_SECONDARY,
                         spikedash="dot", row=1, col=1)
        fig.update_annotations(selector=dict(text="Total portfolio value"), font=dict(color=TEXT_SECONDARY, size=13))
        fig.update_annotations(selector=dict(text="Annual dividend income"), font=dict(color=TEXT_SECONDARY, size=13))

        drip_edge = end["Value_B"] - end["Value_A"]
        subtitle = (f"{self.projection_years} yrs · price +{self.price_growth:.0%}/yr · dividends +{self.dividend_growth:.0%}/yr"
                    f" · DRIP advantage ${drip_edge:,.0f} · Yr-{self.projection_years} income "
                    f"${yearly['Income_B'].iloc[-1]:,.0f} (DRIP) vs ${yearly['Income_A'].iloc[-1]:,.0f} (cash)")
        fig = self._base_layout(fig, f"{self.projection_years}-Year Compound Projection: Cash Payout vs. DRIP",
                                subtitle, height=760)
        fig.update_layout(margin=dict(r=110))
        return fig

    # ----------------------------- output --------------------------------- #

    def export(self, output_dir: str | Path, show: bool = False) -> dict[str, Path]:
        """Write CSVs, individual chart HTML files and a combined dashboard."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary = self.summary if self.summary is not None else self.compute_metrics()
        projection = self.projection if self.projection is not None else self.project()

        paths = {
            "summary_csv": out / "portfolio_summary.csv",
            "schedule_csv": out / "dividend_schedule_12m.csv",
            "projection_csv": out / "projection_monthly.csv",
            "ladder": out / "chart1_dividend_ladder.html",
            "allocation": out / "chart2_weight_vs_income.html",
            "projection": out / "chart3_projection.html",
            "dashboard": out / "dashboard.html",
        }
        summary.to_csv(paths["summary_csv"], index=False)
        self.monthly_schedule().to_csv(paths["schedule_csv"], index_label="Month")
        projection.to_csv(paths["projection_csv"])

        figs = {
            "ladder": self.chart_dividend_ladder(),
            "allocation": self.chart_weight_vs_income(),
            "projection": self.chart_projection(),
        }
        for key, fig in figs.items():
            fig.write_html(paths[key], include_plotlyjs="cdn")

        paths["dashboard"].write_text(self._dashboard_html(figs), encoding="utf-8")
        if show:
            webbrowser.open(paths["dashboard"].resolve().as_uri())
        return paths

    def _dashboard_html(self, figs: dict[str, go.Figure]) -> str:
        """One self-contained page: KPI tiles, metrics table and all three charts."""
        total = self.summary[self.summary["Ticker"] == "TOTAL"].iloc[0]
        tiles = [
            ("Market value", _fmt_money(total["Market Value"])),
            ("Cost basis", _fmt_money(total["Cost Basis"])),
            ("Unrealized G/L", f"{_fmt_money(total['Unrealized G/L'])} ({_fmt_pct(total['G/L %'])})"),
            ("Annual income", _fmt_money(total["Annual Income"])),
            ("Current yield", _fmt_pct(total["Current Yield"])),
            ("Yield on cost", _fmt_pct(total["Yield on Cost"])),
        ]
        tiles_html = "".join(f"<div class='tile'><div class='k'>{k}</div><div class='v'>{v}</div></div>"
                             for k, v in tiles)

        cols = ["Ticker", "Type", "Shares", "Price", "Market Value", "Annual Income", "Current Yield",
                "Yield on Cost", "Payout Ratio", "Safety Flag", "Div CAGR 5Y", "Weight %", "Income %", "Pay Months"]
        table = self.summary[cols].copy()
        for c in ("Price", "Market Value", "Annual Income"):
            table[c] = table[c].map(_fmt_money)
        for c in ("Current Yield", "Yield on Cost", "Payout Ratio", "Div CAGR 5Y", "Weight %", "Income %"):
            table[c] = table[c].map(_fmt_pct)
        table["Shares"] = table["Shares"].map(lambda x: "" if pd.isna(x) else f"{x:,.4g}")
        table["Safety Flag"] = table["Safety Flag"].map(
            lambda f: f"<span class='risky'>⚠ {f}</span>" if str(f).startswith("RISKY")
            else (f"<span class='safe'>✓ {f}</span>" if f == "SAFE" else f))
        table_html = table.to_html(index=False, escape=False, border=0, classes="metrics", na_rep="—")

        chart_html = "".join(
            f"<section class='card'>{fig.to_html(full_html=False, include_plotlyjs=('cdn' if i == 0 else False))}</section>"
            for i, fig in enumerate(figs.values()))
        stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dividend Growth Dashboard</title>
<style>
  :root {{ --surface:{SURFACE}; --ink:{TEXT_PRIMARY}; --ink2:{TEXT_SECONDARY}; --grid:{GRID}; }}
  body {{ margin:0; background:#f3f2ee; color:var(--ink); font-family:{FONT_FAMILY}; }}
  main {{ max-width:1180px; margin:0 auto; padding:24px 16px 48px; }}
  h1 {{ font-size:26px; margin:0 0 4px; }}
  .sub {{ color:var(--ink2); margin:0 0 20px; font-size:14px; }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-bottom:20px; }}
  .tile, .card {{ background:var(--surface); border:1px solid var(--grid); border-radius:10px; }}
  .tile {{ padding:14px 16px; }}
  .k {{ color:var(--ink2); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
  .v {{ font-size:20px; font-weight:600; margin-top:4px; font-variant-numeric:tabular-nums; }}
  .card {{ padding:8px; margin-bottom:20px; overflow-x:auto; }}
  table.metrics {{ border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }}
  table.metrics th {{ text-align:right; color:var(--ink2); font-weight:600; padding:8px 10px; border-bottom:1px solid var(--grid); white-space:nowrap; }}
  table.metrics td {{ text-align:right; padding:7px 10px; border-bottom:1px solid var(--grid); white-space:nowrap; }}
  table.metrics th:first-child, table.metrics td:first-child {{ text-align:left; font-weight:600; }}
  table.metrics tr:last-child td {{ font-weight:700; border-top:2px solid var(--ink2); }}
  .risky {{ color:#b42323; font-weight:600; }} .safe {{ color:#006300; }}
  footer {{ color:var(--ink2); font-size:12px; }}
</style></head>
<body><main>
<h1>Dividend Growth Dashboard</h1>
<p class="sub">Data from Yahoo Finance as of {stamp}. Estimates only — not investment advice.</p>
<div class="tiles">{tiles_html}</div>
<section class="card">{table_html}</section>
{chart_html}
<footer>Payout limits: {CORPORATE_PAYOUT_LIMIT:.0%} corporations, {REIT_PAYOUT_LIMIT:.0%} REITs (GAAP EPS basis; REITs are better judged on AFFO).
Projection assumes {self.price_growth:.0%} annual price growth and {self.dividend_growth:.0%} annual dividend growth, no taxes or new contributions.</footer>
</main></body></html>"""

    # ----------------------------- pipeline ------------------------------- #

    def run(self, output_dir: str | Path = "output", show: bool = False) -> dict[str, Path]:
        """Full pipeline: fetch -> metrics -> projection -> charts/exports."""
        self.fetch_data()
        self.compute_metrics()
        self.project()
        self.print_summary()

        proj = self.projection
        print(f"\n{self.projection_years}-YEAR PROJECTION (price +{self.price_growth:.0%}/yr, "
              f"dividends +{self.dividend_growth:.0%}/yr)")
        print(f"  A · Cash payout : ${proj['Value_A'].iloc[-1]:,.0f}  (of which cash collected ${proj['Cash_A'].iloc[-1]:,.0f})")
        print(f"  B · Full DRIP   : ${proj['Value_B'].iloc[-1]:,.0f}  (share count ×{proj['Shares_B_Factor'].iloc[-1]:.2f})")
        print(f"  DRIP advantage  : ${proj['Value_B'].iloc[-1] - proj['Value_A'].iloc[-1]:,.0f}")

        paths = self.export(output_dir, show=show)
        print("\nFiles written:")
        for path in paths.values():
            print(f"  {path}")
        return paths


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dividend Growth Tracker & Visualizer")
    p.add_argument("--portfolio", type=Path, help="CSV with columns Ticker,Shares,AvgCost (default: sample portfolio)")
    p.add_argument("--output-dir", type=Path, default=Path("output"), help="where to write charts and CSVs")
    p.add_argument("--years", type=int, default=10, help="projection horizon in years (default 10)")
    p.add_argument("--price-growth", type=float, default=0.05, help="annual price appreciation (default 0.05)")
    p.add_argument("--div-growth", type=float, default=0.06, help="annual dividend growth (default 0.06)")
    p.add_argument("--pay-lag-days", type=int, default=0,
                   help="days added to ex-dividend dates to approximate pay dates (default 0)")
    p.add_argument("--concentration-limit", type=float, default=INCOME_CONCENTRATION_LIMIT,
                   help="flag positions above this share of total income (default 0.25)")
    p.add_argument("--show", action="store_true", help="open the dashboard in a browser when done")
    p.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    # Silence pandas deprecation chatter raised from inside yfinance itself.
    warnings.filterwarnings("ignore", module="yfinance")

    portfolio = pd.read_csv(args.portfolio) if args.portfolio else SAMPLE_PORTFOLIO
    try:
        tracker = PortfolioTracker(
            portfolio,
            price_growth=args.price_growth,
            dividend_growth=args.div_growth,
            projection_years=args.years,
            pay_lag_days=args.pay_lag_days,
            concentration_limit=args.concentration_limit,
        )
        tracker.run(output_dir=args.output_dir, show=args.show)
    except (ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
