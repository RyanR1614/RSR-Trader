# RSR — Resilient Signal & Returns
### An AI/ML Paper Trading Bot

RSR fetches stock price and news data, engineers technical + sentiment features,
trains an LSTM neural network to predict price direction, and simulates buy/sell/hold
decisions on a virtual $100,000 portfolio — all running automatically in the cloud.

---

## Quickstart (Local)

### 1. Clone & set up environment

```bash
git clone https://github.com/YOUR_USERNAME/rsr_trading_bot.git
cd rsr_trading_bot

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download NLP models (one-time)

```bash
python -m nltk.downloader stopwords wordnet omw-1.4 vader_lexicon
python -m spacy download en_core_web_sm
```

### 3. Set your API keys

```bash
cp .env.example .env
# Edit .env and fill in your keys (NewsAPI is free at newsapi.org)
# yfinance (price data) needs NO key at all
```

Then load the variables:
```bash
# macOS/Linux
export $(cat .env | xargs)

# Windows PowerShell
Get-Content .env | ForEach-Object { $key, $val = $_ -split '=', 2; [System.Environment]::SetEnvironmentVariable($key, $val) }
```

### 4. Download historical price data

```bash
python data/fetch_prices.py
# Downloads 2 years of OHLCV data for all tickers in config/settings.py
# Saves to data/raw/
```

### 5. Train the model

```bash
python pipeline/train.py
# Trains an LSTM model for each ticker (~2-5 min per ticker on CPU)
# Saves best model weights to models/saved/
# You need to do this before running the daily pipeline
```

### 6. Run one daily cycle (manually)

```bash
python pipeline/daily_run.py
# Fetches latest data, runs model inference, executes simulated trades
# Saves portfolio state to data/portfolio.json
# Generates charts to plots/
```

### 7. View performance charts

```bash
python plots/generate_charts.py
# Generates/updates plots/portfolio.png
# Open the plots/ folder to see charts
```

### 8. Run a backtest

```bash
python trading/backtest.py
# Walk-forward backtest on historical data
# Prints Sharpe ratio, max drawdown, total return
# Saves charts to plots/
```

---

## Project Structure

```
rsr_trading_bot/
├── config/
│   └── settings.py          # All configuration — tickers, hyperparameters, paths
├── data/
│   ├── fetch_prices.py      # Download price data via yfinance
│   ├── raw/                 # Downloaded OHLCV CSVs (git-ignored)
│   └── sentiment/           # Saved news JSON (git-ignored)
├── features/
│   ├── technical.py         # RSI, MACD, Bollinger Bands, ATR, etc.
│   └── combine.py           # Merge technical + sentiment features
├── sentiment/
│   ├── news_fetcher.py      # NewsAPI + Yahoo RSS + Reddit PRAW
│   ├── preprocessor.py      # Text cleaning + lemmatization
│   ├── scorer.py            # VADER sentiment scoring (fast, no GPU)
│   └── finbert_scorer.py    # Optional: FinBERT (higher accuracy, needs GPU)
├── models/
│   ├── dataset.py           # PyTorch Dataset with sliding windows
│   ├── price_predictor.py   # LSTM architecture with attention
│   └── saved/               # Trained model weights (git-ignored)
├── trading/
│   ├── portfolio.py         # Virtual cash + holdings tracker
│   ├── strategy.py          # BUY/SELL/HOLD decision logic
│   ├── backtest.py          # Walk-forward backtesting + metrics
│   └── rl_env.py            # Optional: Gymnasium RL environment
├── pipeline/
│   ├── daily_run.py         # Main orchestrator — run this daily
│   ├── train.py             # Model training script
│   ├── train_rl.py          # Optional: PPO reinforcement learning
│   └── scheduler.py         # For Railway/always-on deployment
├── plots/
│   └── generate_charts.py   # Portfolio + signals charts
├── utils/
│   └── rate_limit.py        # Retry decorator for API calls
├── .github/workflows/
│   └── rsr_daily.yml        # GitHub Actions — free cloud scheduling
├── Dockerfile               # For GCP Cloud Run / Railway
├── Procfile                 # For Railway deployment
├── .env.example             # Template for API keys
└── requirements.txt
```

---

## Cloud Deployment (Free — GitHub Actions)

The easiest way to run RSR 24/7 without your laptop.

### Setup
1. Push your repo to GitHub (can be private)
2. Go to: **Settings → Secrets and variables → Actions**
3. Add secrets: `NEWSAPI_KEY`, `REDDIT_CLIENT_ID`, `REDDIT_SECRET`
4. The workflow at `.github/workflows/rsr_daily.yml` runs automatically at **9:35 AM ET on weekdays**
5. After each run, download logs and charts from the **Actions → Artifacts** section

### Alternative: Railway
```bash
# Install Railway CLI
npm install -g @railway/cli
railway login
railway init
railway up
# Set env vars in Railway dashboard
```

---

## Configuration

Edit `config/settings.py` to customize:

| Setting | Default | Description |
|---|---|---|
| `TICKERS` | 6 stocks | Which stocks to trade |
| `STARTING_CASH` | $100,000 | Virtual portfolio size |
| `TRADE_SIZE_PCT` | 5% | Size of each trade |
| `MAX_POSITION_PCT` | 10% | Max exposure per stock |
| `LOOKBACK_DAYS` | 30 | Days of history per model input |
| `EPOCHS` | 50 | Training epochs |
| `BUY_THRESHOLD` | 0.60 | Signal above this → BUY |
| `SELL_THRESHOLD` | 0.40 | Signal below this → SELL |

---

## Optional: Reinforcement Learning

After running the standard pipeline, train a PPO agent:

```bash
pip install stable-baselines3 gymnasium
python pipeline/train_rl.py AAPL
```

---

## Practical Notes

- **This is a paper trading system** — no real money is ever used
- **Past simulated returns do not predict future results**
- yfinance is free and requires no API key
- NewsAPI free tier: 100 requests/day — sufficient for up to ~50 tickers
- The model typically achieves 52–56% directional accuracy (beating random but not by much — stock prediction is hard by design)
- Retrain models weekly or monthly for best results

---

*RSR — Resilient Signal & Returns*
