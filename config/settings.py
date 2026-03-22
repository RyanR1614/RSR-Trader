"""
RSR — Resilient Signal & Returns
Central configuration. All secrets are read from environment variables.
Never hardcode API keys.
"""
import os

# ── API Keys ─────────────────────────────────────────────────────────────────
NEWSAPI_KEY        = os.environ.get("NEWSAPI_KEY", "")
ALPHA_VANTAGE_KEY  = os.environ.get("AV_KEY", "")
REDDIT_CLIENT_ID   = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_SECRET      = os.environ.get("REDDIT_SECRET", "")
REDDIT_USER_AGENT  = "RSR_TradingBot/1.0"

# ── Stocks to Trade ───────────────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]

# ── Portfolio ─────────────────────────────────────────────────────────────────
STARTING_CASH     = 100_000.00   # Virtual starting capital ($)
MAX_POSITION_PCT  = 0.10         # Max 10% of portfolio per stock
TRADE_SIZE_PCT    = 0.05         # 5% of portfolio per trade

# ── Model Hyperparameters ─────────────────────────────────────────────────────
LOOKBACK_DAYS       = 30
PREDICTION_HORIZON  = 1
HIDDEN_SIZE         = 128
NUM_LAYERS          = 2
DROPOUT             = 0.3
LEARNING_RATE       = 0.001
BATCH_SIZE          = 64
EPOCHS              = 50

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR         = "data/"
MODEL_DIR        = "models/saved/"
LOG_FILE         = "logs/rsr.log"
PORTFOLIO_FILE   = "data/portfolio.json"
PLOTS_DIR        = "plots/"
