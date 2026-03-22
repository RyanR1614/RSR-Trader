"""
RSR — Chart Generation
Generates portfolio performance and signal charts.
Call plot_portfolio() after each daily run to keep charts current.
"""
import json
import logging
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from config.settings import PORTFOLIO_FILE, STARTING_CASH, PLOTS_DIR

logger = logging.getLogger("rsr")

# Dark theme used for all charts
DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
BLUE       = "#58a6ff"
GREEN      = "#3fb950"
RED        = "#f78166"
PURPLE     = "#d2a8ff"
GRAY       = "#8b949e"
TEXT_COLOR = "#c9d1d9"


def _apply_dark_style(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(True, alpha=0.15, color="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


def plot_portfolio():
    """
    Generate the main portfolio performance dashboard.
    Reads from data/portfolio.json and writes to plots/portfolio.png.
    """
    if not os.path.exists(PORTFOLIO_FILE):
        logger.warning("No portfolio file found. Run daily_run.py first.")
        return

    with open(PORTFOLIO_FILE) as f:
        data = json.load(f)

    history = data.get("history", [])
    trades  = data.get("trades", [])

    if len(history) < 2:
        logger.info("Not enough history to plot yet (need at least 2 snapshots).")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.DataFrame(history)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), facecolor=DARK_BG)
    fig.suptitle("RSR — Portfolio Performance Dashboard", color="white", fontsize=14,
                 fontweight="bold", y=0.99)

    dates = df.index

    # ── Panel 1: Total portfolio value ────────────────────────────────────────
    ax1 = axes[0]
    _apply_dark_style(ax1)
    ax1.plot(dates, df["total"], color=BLUE, linewidth=2, label="Total Value")
    ax1.plot(dates, df["cash"],  color=GREEN, linewidth=1, linestyle="--",
             alpha=0.7, label="Cash")
    ax1.fill_between(dates, df["holdings_value"], 0,
                     alpha=0.2, color=RED, label="Stocks")
    ax1.axhline(y=STARTING_CASH, color=GRAY, linestyle=":", alpha=0.6,
                label=f"Start ${STARTING_CASH:,.0f}")

    # Mark trade dates
    if trades:
        tdf = pd.DataFrame(trades)
        tdf["date"] = pd.to_datetime(tdf["date"])
        buys  = tdf[tdf["action"] == "BUY"]
        sells = tdf[tdf["action"] == "SELL"]
        for _, row in buys.iterrows():
            if row["date"] in df.index:
                ax1.axvline(row["date"], color=GREEN, alpha=0.3, linewidth=1)
        for _, row in sells.iterrows():
            if row["date"] in df.index:
                ax1.axvline(row["date"], color=RED, alpha=0.3, linewidth=1)

    ax1.set_ylabel("Value ($)")
    ax1.legend(facecolor="#21262d", labelcolor="white", fontsize=9, loc="upper left")

    # Annotate current value
    current = df["total"].iloc[-1]
    ret_pct = (current / STARTING_CASH - 1) * 100
    ax1.annotate(
        f"${current:,.0f}  ({ret_pct:+.1f}%)",
        xy=(dates[-1], current),
        xytext=(-80, 15), textcoords="offset points",
        color=GREEN if ret_pct >= 0 else RED,
        fontsize=10, fontweight="bold",
    )

    # ── Panel 2: Daily return % ───────────────────────────────────────────────
    ax2 = axes[1]
    _apply_dark_style(ax2)
    daily_ret = df["total"].pct_change() * 100
    colors    = [GREEN if r >= 0 else RED for r in daily_ret]
    ax2.bar(dates, daily_ret, color=colors, alpha=0.8, width=0.8)
    ax2.axhline(y=0, color=GRAY, linewidth=0.8)
    ax2.set_ylabel("Daily Return (%)")

    # Rolling mean
    rolling_mean = daily_ret.rolling(10).mean()
    ax2.plot(dates, rolling_mean, color=BLUE, linewidth=1.5,
             alpha=0.8, label="10-day avg")
    ax2.legend(facecolor="#21262d", labelcolor="white", fontsize=9)

    # ── Panel 3: Cumulative return vs buy-and-hold ────────────────────────────
    ax3 = axes[2]
    _apply_dark_style(ax3)
    cum_return = (df["total"] / STARTING_CASH - 1) * 100

    # Max drawdown shading
    running_max = df["total"].cummax()
    drawdown    = (df["total"] - running_max) / (running_max + 1e-9) * 100
    ax3.fill_between(dates, drawdown, 0, alpha=0.3, color=RED, label="Drawdown")
    ax3.plot(dates, cum_return, color=BLUE, linewidth=2, label="RSR Return %")
    ax3.axhline(y=0, color=GRAY, linewidth=0.8)
    ax3.set_ylabel("Cumulative Return (%)")
    ax3.set_xlabel("Date")
    ax3.legend(facecolor="#21262d", labelcolor="white", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = f"{PLOTS_DIR}portfolio.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    logger.info(f"Portfolio chart saved → {out_path}")


def plot_signals(signals: dict):
    """
    Bar chart of today's model signals and sentiment per ticker.
    signals: {ticker: {prob_up, sentiment, price, action}}
    """
    if not signals:
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)

    tickers   = list(signals.keys())
    prob_ups  = [signals[t]["prob_up"]   for t in tickers]
    sentiments = [signals[t]["sentiment"] for t in tickers]
    actions   = [signals[t]["action"]    for t in tickers]

    action_colors = {
        "BUY": GREEN, "SELL": RED, "HOLD": GRAY, "STOP_LOSS": "#ff6b35"
    }
    bar_colors = [action_colors.get(a, GRAY) for a in actions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)
    fig.suptitle("RSR — Today's Signals", color="white", fontsize=13, fontweight="bold")

    # P(up) bars
    _apply_dark_style(ax1)
    bars = ax1.bar(tickers, prob_ups, color=bar_colors, alpha=0.85)
    ax1.axhline(y=0.6, color=GREEN, linestyle="--", alpha=0.5, linewidth=1.2, label="Buy (0.6)")
    ax1.axhline(y=0.4, color=RED,   linestyle="--", alpha=0.5, linewidth=1.2, label="Sell (0.4)")
    ax1.axhline(y=0.5, color=GRAY,  linestyle=":",  alpha=0.4, linewidth=1.0)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("P(price up)")
    ax1.set_title("Model Signal", color=TEXT_COLOR)
    ax1.legend(facecolor="#21262d", labelcolor="white", fontsize=9)
    for bar, val, action in zip(bars, prob_ups, actions):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.2f}\n{action}", ha="center", va="bottom",
                 color="white", fontsize=8, fontweight="bold")

    # Sentiment bars
    _apply_dark_style(ax2)
    sent_colors = [GREEN if s >= 0 else RED for s in sentiments]
    ax2.bar(tickers, sentiments, color=sent_colors, alpha=0.85)
    ax2.axhline(y=0, color=GRAY, linewidth=0.8)
    ax2.set_ylim(-1, 1)
    ax2.set_ylabel("Sentiment Score")
    ax2.set_title("News Sentiment (compound)", color=TEXT_COLOR)
    for i, (t, s) in enumerate(zip(tickers, sentiments)):
        ax2.text(i, s + (0.03 if s >= 0 else -0.06), f"{s:+.2f}",
                 ha="center", color="white", fontsize=9)

    plt.tight_layout()
    out_path = f"{PLOTS_DIR}signals_today.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    logger.info(f"Signals chart saved → {out_path}")


def print_portfolio_table():
    """Print a text-based portfolio summary to the console."""
    if not os.path.exists(PORTFOLIO_FILE):
        print("No portfolio file. Run pipeline/daily_run.py first.")
        return
    with open(PORTFOLIO_FILE) as f:
        data = json.load(f)
    history = data.get("history", [])
    if not history:
        print("No portfolio history yet.")
        return
    latest = history[-1]
    print("\n" + "=" * 45)
    print("  RSR Portfolio — Latest Snapshot")
    print("=" * 45)
    print(f"  Date:            {latest['date']}")
    print(f"  Cash:            ${latest['cash']:>12,.2f}")
    print(f"  Holdings value:  ${latest['holdings_value']:>12,.2f}")
    print(f"  Total value:     ${latest['total']:>12,.2f}")
    print(f"  Return:          {latest['return_pct']:>+11.2f}%")
    print("=" * 45)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    plot_portfolio()
    print_portfolio_table()
