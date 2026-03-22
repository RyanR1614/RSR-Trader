"""
RSR — Data Acquisition: Historical Stock Prices via yfinance
No API key required. Run standalone to download/refresh all tickers.
"""
import os
import logging
import yfinance as yf
import pandas as pd

from config.settings import TICKERS, DATA_DIR

logger = logging.getLogger("rsr")


def download_prices(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Download historical OHLCV data for a ticker.
    period: "1y", "2y", "5y", "max"
    Returns DataFrame indexed by date with columns: Open, High, Low, Close, Volume
    """
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    # Flatten MultiIndex columns if present (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def update_all_tickers(period: str = "2y") -> dict[str, pd.DataFrame]:
    """Download and save price data for every ticker in settings."""
    os.makedirs(DATA_DIR + "raw", exist_ok=True)
    results = {}
    for ticker in TICKERS:
        logger.info(f"Fetching price data for {ticker}...")
        try:
            df = download_prices(ticker, period)
            path = f"{DATA_DIR}raw/{ticker}_prices.csv"
            df.to_csv(path)
            logger.info(f"  Saved {len(df)} rows → {path}")
            results[ticker] = df
        except Exception as e:
            logger.error(f"  Failed to fetch {ticker}: {e}")
    return results


def load_prices(ticker: str) -> pd.DataFrame:
    """Load saved price CSV for a ticker."""
    path = f"{DATA_DIR}raw/{ticker}_prices.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No price data found for {ticker}. Run update_all_tickers() first.")
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    update_all_tickers()
