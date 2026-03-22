"""
RSR — Model Training Pipeline
Trains an LSTM model for each ticker and saves the best weights.
Run this once before the first daily_run.py, then periodically (e.g. weekly).

Usage:
    python pipeline/train.py              # train all tickers
    python pipeline/train.py AAPL MSFT   # train specific tickers
"""
import logging
import os
import sys

import torch
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    TICKERS, DATA_DIR, MODEL_DIR, LOG_FILE,
    LOOKBACK_DAYS, HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
    LEARNING_RATE, BATCH_SIZE, EPOCHS,
)
from data.fetch_prices import load_prices
from features.combine import build_full_feature_set, ALL_FEATURE_COLS
from models.dataset import StockDataset
from models.price_predictor import RSRPredictor

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


def train_ticker(ticker: str) -> RSRPredictor:
    """Full training run for one ticker. Returns trained model."""
    logger.info(f"{'='*50}")
    logger.info(f"Training model for {ticker}")

    # Load and prepare data
    df_raw = load_prices(ticker)
    df     = build_full_feature_set(ticker, df_raw)

    if len(df) < LOOKBACK_DAYS + 50:
        logger.warning(f"Not enough data for {ticker} ({len(df)} rows). Skipping.")
        return None

    # Create dataset (fit scaler on full data for training)
    dataset      = StockDataset(df, ALL_FEATURE_COLS, LOOKBACK_DAYS)
    num_features = len(ALL_FEATURE_COLS)

    # Chronological 80/20 split — no shuffle on split
    n_train = int(0.80 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds = Subset(dataset, range(0, n_train))
    val_ds   = Subset(dataset, range(n_train, len(dataset)))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # Model, optimizer, loss
    model     = RSRPredictor(num_features, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=7, factor=0.5, verbose=False
    )

    best_val_loss  = float("inf")
    patience_count = 0
    early_stop     = 15   # stop if val loss doesn't improve for 15 epochs
    os.makedirs(MODEL_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss, n_batches = 0.0, 0
        for X, y in train_loader:
            optimizer.zero_grad()
            preds = model(X)
            loss  = criterion(preds, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches  += 1
        train_loss /= max(n_batches, 1)

        # ── Validate ──
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                preds     = model(X)
                val_loss += criterion(preds, y).item()
                correct  += ((preds > 0.5).float() == y).sum().item()
                total    += len(y)
        val_loss /= max(len(val_loader), 1)
        accuracy  = correct / max(total, 1) * 100

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:3d}/{EPOCHS}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"acc={accuracy:.1f}%"
            )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            save_path = f"{MODEL_DIR}{ticker}_best.pt"
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "val_loss":     val_loss,
                "val_accuracy": accuracy,
                "num_features": num_features,
                "feature_cols": ALL_FEATURE_COLS,
                "hidden_size":  HIDDEN_SIZE,
                "num_layers":   NUM_LAYERS,
            }, save_path)
        else:
            patience_count += 1
            if patience_count >= early_stop:
                logger.info(f"  Early stopping at epoch {epoch}.")
                break

    logger.info(
        f"Training complete for {ticker}. "
        f"Best val loss: {best_val_loss:.4f}"
    )

    # Load best weights back into model
    checkpoint = torch.load(f"{MODEL_DIR}{ticker}_best.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    return model


def train_all(tickers: list[str] = None):
    tickers = tickers or TICKERS
    results = {}
    for ticker in tickers:
        try:
            model = train_ticker(ticker)
            results[ticker] = "OK" if model else "SKIPPED"
        except FileNotFoundError:
            logger.error(f"No price data for {ticker}. Run: python data/fetch_prices.py")
            results[ticker] = "NO_DATA"
        except Exception as e:
            logger.error(f"Training failed for {ticker}: {e}", exc_info=True)
            results[ticker] = f"ERROR: {e}"

    logger.info("Training summary:")
    for t, r in results.items():
        logger.info(f"  {t:8}: {r}")
    return results


if __name__ == "__main__":
    tickers = sys.argv[1:] if len(sys.argv) > 1 else None
    train_all(tickers)
