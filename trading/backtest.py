"""
RSR — Backtesting Engine
Walk-forward backtest: trains only on past data at each step (no lookahead bias).
Computes Sharpe ratio, max drawdown, total return, and generates charts.
"""
import logging
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from torch import nn, optim
from torch.utils.data import DataLoader

from config.settings import (
    LOOKBACK_DAYS, HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
    LEARNING_RATE, BATCH_SIZE, STARTING_CASH, PLOTS_DIR,
)
from features.combine import build_full_feature_set, ALL_FEATURE_COLS
from models.dataset import StockDataset
from models.price_predictor import RSRPredictor
from trading.portfolio import Portfolio
from trading.strategy import make_decision

logger = logging.getLogger("rsr")


# ── Walk-Forward Backtest ─────────────────────────────────────────────────────

def run_backtest(ticker: str, df_raw: pd.DataFrame, retrain_every: int = 20) -> pd.DataFrame:
    """
    Walk-forward backtest for a single ticker.
    Uses first 70% of data for initial training, then steps forward day by day.
    Retrains model every `retrain_every` steps to stay current.

    Returns a DataFrame with columns: [value, cash, action] indexed by date.
    """
    logger.info(f"Starting backtest for {ticker}...")
    df           = build_full_feature_set(ticker, df_raw)
    dates        = df.index.tolist()
    split_idx    = int(0.70 * len(df))

    portfolio    = Portfolio()
    portfolio.reset()
    portfolio.cash = STARTING_CASH

    model        = None
    scaler       = None
    value_records = []

    for i in range(split_idx, len(df) - 1):
        train_df  = df.iloc[:i]
        price     = float(df.iloc[i]["Close"])
        today     = dates[i]

        # (Re)train on available data
        if model is None or (i - split_idx) % retrain_every == 0:
            model, scaler = _train_model(train_df)
            if model is None:
                continue

        # Predict
        model.eval()
        try:
            last_window = torch.tensor(
                scaler.transform(train_df[ALL_FEATURE_COLS].values[-LOOKBACK_DAYS:]),
                dtype=torch.float32,
            ).unsqueeze(0)
            with torch.no_grad():
                prob_up = model(last_window).item()
        except Exception as e:
            logger.warning(f"Prediction failed on {today}: {e}")
            prob_up = 0.5

        sentiment = float(df.iloc[i].get("sentiment_compound", 0.0))
        prices    = {ticker: price}
        total_val = portfolio.total_value(prices)
        action    = make_decision(ticker, price, prob_up, sentiment, portfolio, total_val)

        value_records.append({
            "date":   today,
            "value":  portfolio.total_value(prices),
            "cash":   portfolio.cash,
            "action": action,
            "price":  price,
            "prob_up": round(prob_up, 4),
        })

    result_df = pd.DataFrame(value_records).set_index("date")
    logger.info(f"Backtest complete for {ticker}. {len(result_df)} trading days evaluated.")
    return result_df


def _train_model(train_df: pd.DataFrame):
    """Quick model train for walk-forward use."""
    try:
        dataset = StockDataset(train_df, ALL_FEATURE_COLS, LOOKBACK_DAYS)
        if len(dataset) < 20:
            return None, None
        loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        model     = RSRPredictor(len(ALL_FEATURE_COLS), HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        criterion = nn.BCELoss()
        model.train()
        for _ in range(15):   # quick train — full training done in pipeline/train.py
            for X, y in loader:
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        return model, dataset.scaler
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return None, None


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(value_df: pd.DataFrame) -> dict:
    """
    Compute standard performance metrics from a backtest result DataFrame.
    """
    values  = value_df["value"]
    returns = values.pct_change().dropna()

    total_return  = (values.iloc[-1] / values.iloc[0] - 1) * 100
    sharpe        = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
    running_max   = values.cummax()
    drawdown      = (values - running_max) / (running_max + 1e-9)
    max_drawdown  = drawdown.min() * 100
    win_days      = (returns > 0).sum()
    total_days    = len(returns)
    win_rate      = (win_days / total_days * 100) if total_days > 0 else 0

    trades = value_df[value_df["action"].isin(["BUY", "SELL"])]

    return {
        "total_return_pct":  round(total_return, 2),
        "sharpe_ratio":      round(float(sharpe), 3),
        "max_drawdown_pct":  round(float(max_drawdown), 2),
        "win_rate_pct":      round(win_rate, 2),
        "final_value":       round(float(values.iloc[-1]), 2),
        "starting_value":    round(float(values.iloc[0]), 2),
        "total_trades":      len(trades),
        "trading_days":      total_days,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_backtest(value_df: pd.DataFrame, ticker: str, metrics: dict):
    """Generate a 3-panel backtest chart and save to plots/."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), facecolor="#0d1117")
    fig.suptitle(
        f"RSR Backtest — {ticker}   |   "
        f"Return: {metrics['total_return_pct']:+.1f}%   "
        f"Sharpe: {metrics['sharpe_ratio']:.2f}   "
        f"Max DD: {metrics['max_drawdown_pct']:.1f}%",
        color="white", fontsize=13, fontweight="bold", y=0.98,
    )

    dates = pd.to_datetime(value_df.index)

    # ── Panel 1: Portfolio value ──────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#161b22")
    ax1.plot(dates, value_df["value"], color="#58a6ff", linewidth=2, label="Portfolio Value")
    ax1.axhline(y=STARTING_CASH, color="#8b949e", linestyle="--", alpha=0.6, label=f"Start ${STARTING_CASH:,.0f}")

    # Mark buy/sell points
    buys  = value_df[value_df["action"] == "BUY"]
    sells = value_df[value_df["action"] == "SELL"]
    ax1.scatter(pd.to_datetime(buys.index),  buys["value"],  color="#3fb950", s=30, zorder=5, label="BUY",  marker="^")
    ax1.scatter(pd.to_datetime(sells.index), sells["value"], color="#f78166", s=30, zorder=5, label="SELL", marker="v")

    ax1.set_ylabel("Portfolio Value ($)", color="#c9d1d9")
    ax1.tick_params(colors="#c9d1d9")
    ax1.legend(facecolor="#21262d", labelcolor="white", fontsize=9)
    ax1.grid(True, alpha=0.15, color="white")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    running_max = value_df["value"].cummax()
    drawdown    = (value_df["value"] - running_max) / (running_max + 1e-9) * 100
    ax2.fill_between(dates, drawdown, 0, color="#f78166", alpha=0.5)
    ax2.plot(dates, drawdown, color="#f78166", linewidth=1)
    ax2.set_ylabel("Drawdown (%)", color="#c9d1d9")
    ax2.tick_params(colors="#c9d1d9")
    ax2.grid(True, alpha=0.15, color="white")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    # ── Panel 3: Model P(up) signal ───────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor("#161b22")
    ax3.plot(dates, value_df["prob_up"], color="#d2a8ff", linewidth=1, alpha=0.8)
    ax3.axhline(y=0.6, color="#3fb950", linestyle="--", alpha=0.5, linewidth=1, label="Buy threshold")
    ax3.axhline(y=0.4, color="#f78166", linestyle="--", alpha=0.5, linewidth=1, label="Sell threshold")
    ax3.axhline(y=0.5, color="#8b949e", linestyle=":",  alpha=0.4, linewidth=1)
    ax3.set_ylabel("P(price up)", color="#c9d1d9")
    ax3.set_xlabel("Date", color="#c9d1d9")
    ax3.tick_params(colors="#c9d1d9")
    ax3.legend(facecolor="#21262d", labelcolor="white", fontsize=9)
    ax3.grid(True, alpha=0.15, color="white")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax3.set_ylim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = f"{PLOTS_DIR}{ticker}_backtest.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    logger.info(f"Backtest chart saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from data.fetch_prices import load_prices
    from config.settings import TICKERS

    for ticker in TICKERS[:2]:   # run first 2 tickers by default
        try:
            df_raw  = load_prices(ticker)
            results = run_backtest(ticker, df_raw)
            metrics = compute_metrics(results)
            print(f"\n{ticker} Backtest Metrics:")
            for k, v in metrics.items():
                print(f"  {k:<22}: {v}")
            plot_backtest(results, ticker, metrics)
        except FileNotFoundError:
            print(f"No data for {ticker}. Run: python data/fetch_prices.py")
        except Exception as e:
            print(f"Backtest failed for {ticker}: {e}")
