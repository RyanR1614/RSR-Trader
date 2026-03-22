"""
RSR — Daily Pipeline Orchestrator
This is the main script executed by cron / GitHub Actions / Railway every trading day.
It fetches data, generates predictions, executes simulated trades, logs everything,
and saves an updated portfolio snapshot + performance chart.

Usage:
    python pipeline/daily_run.py
"""
import logging
import os
import sys

import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    TICKERS, DATA_DIR, MODEL_DIR, LOG_FILE, LOOKBACK_DAYS,
    HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
)
from data.fetch_prices import download_prices
from sentiment.news_fetcher import fetch_all_news
from features.combine import build_full_feature_set, ALL_FEATURE_COLS
from models.dataset import StockDataset
from models.price_predictor import RSRPredictor
from trading.portfolio import Portfolio
from trading.strategy import make_decision
from plots.generate_charts import plot_portfolio, plot_signals

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("rsr")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ticker: str, num_features: int) -> tuple:
    """
    Load saved model + scaler for a ticker.
    Returns (model, scaler) or (None, None) if no saved model.
    """
    model_path = f"{MODEL_DIR}{ticker}_best.pt"
    if not os.path.exists(model_path):
        logger.warning(f"  No saved model for {ticker}. Predictions will be random.")
        return None, None
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        model = RSRPredictor(
            num_features = checkpoint.get("num_features", num_features),
            hidden_size  = checkpoint.get("hidden_size", HIDDEN_SIZE),
            num_layers   = checkpoint.get("num_layers",  NUM_LAYERS),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        logger.info(
            f"  Loaded model for {ticker} "
            f"(val_loss={checkpoint.get('val_loss', '?'):.4f}, "
            f"acc={checkpoint.get('val_accuracy', '?'):.1f}%)"
        )
        return model, None   # scaler rebuilt from today's data below
    except Exception as e:
        logger.error(f"  Failed to load model for {ticker}: {e}")
        return None, None


def predict(model: RSRPredictor, dataset: StockDataset) -> float:
    """Run inference and return P(price up tomorrow) in [0, 1]."""
    if model is None:
        return 0.5   # neutral if no model
    try:
        model.eval()
        window = dataset.get_last_window()
        with torch.no_grad():
            prob = model(window).item()
        return float(prob)
    except Exception as e:
        logger.warning(f"  Prediction failed: {e}")
        return 0.5


# ── Main daily run ────────────────────────────────────────────────────────────

def run_daily():
    logger.info("=" * 60)
    logger.info("RSR — Daily Run Starting")
    logger.info("=" * 60)

    portfolio   = Portfolio()
    prices_now  = {}
    signals     = {}   # {ticker: {prob_up, sentiment, action}}

    for ticker in TICKERS:
        logger.info(f"\n--- {ticker} ---")

        # ── 1. Update price data ──────────────────────────────────────────────
        try:
            df_raw = download_prices(ticker, period="2y")
            os.makedirs(f"{DATA_DIR}raw", exist_ok=True)
            df_raw.to_csv(f"{DATA_DIR}raw/{ticker}_prices.csv")
        except Exception as e:
            logger.error(f"  Price fetch failed: {e}")
            continue

        # ── 2. Fetch news & sentiment ─────────────────────────────────────────
        try:
            fetch_all_news(ticker)
        except Exception as e:
            logger.warning(f"  News fetch failed (continuing without): {e}")

        # ── 3. Build features ─────────────────────────────────────────────────
        try:
            df = build_full_feature_set(ticker, df_raw)
        except Exception as e:
            logger.error(f"  Feature engineering failed: {e}")
            continue

        if len(df) < LOOKBACK_DAYS + 5:
            logger.warning(f"  Not enough data rows ({len(df)}). Skipping.")
            continue

        # ── 4. Build dataset (for scaler) ─────────────────────────────────────
        try:
            dataset = StockDataset(df, ALL_FEATURE_COLS, LOOKBACK_DAYS)
        except Exception as e:
            logger.error(f"  Dataset build failed: {e}")
            continue

        # ── 5. Load model & predict ───────────────────────────────────────────
        model, _  = load_model(ticker, len(ALL_FEATURE_COLS))
        prob_up   = predict(model, dataset)

        # ── 6. Get current price & sentiment ─────────────────────────────────
        current_price = float(df["Close"].iloc[-1])
        sentiment     = float(df["sentiment_compound"].iloc[-1])
        prices_now[ticker] = current_price

        logger.info(
            f"  Price=${current_price:.2f}  "
            f"P(up)={prob_up:.3f}  "
            f"Sentiment={sentiment:+.3f}"
        )

        # ── 7. Make trading decision ──────────────────────────────────────────
        total_val = portfolio.total_value(prices_now)
        action    = make_decision(
            ticker, current_price, prob_up, sentiment, portfolio, total_val
        )
        logger.info(f"  Decision: {action}")

        signals[ticker] = {
            "prob_up":   round(prob_up, 4),
            "sentiment": round(sentiment, 4),
            "price":     round(current_price, 2),
            "action":    action,
        }

    # ── 8. Record snapshot & save ─────────────────────────────────────────────
    if prices_now:
        portfolio.record_snapshot(prices_now)
        portfolio.save()
        logger.info("\n" + portfolio.summary(prices_now))
    else:
        logger.warning("No tickers processed — portfolio not updated.")

    # ── 9. Generate updated charts ────────────────────────────────────────────
    try:
        plot_portfolio()
        if signals:
            plot_signals(signals)
    except Exception as e:
        logger.warning(f"Chart generation failed: {e}")

    logger.info("=" * 60)
    logger.info("RSR — Daily Run Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_daily()
