"""
RSR — Feature Engineering: Technical Indicators
Transforms raw OHLCV price data into model-ready features.
"""
import pandas as pd
import numpy as np


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  DataFrame with columns [Open, High, Low, Close, Volume]
    Output: Same DataFrame with added technical indicator columns.
    Drops rows with NaN (warmup period).
    """
    c = df["Close"].copy()

    # ── Returns ──────────────────────────────────────────────────────────────
    df["daily_return"] = c.pct_change()
    df["log_return"]   = np.log(c / c.shift(1))

    # ── Simple & Exponential Moving Averages ─────────────────────────────────
    df["sma_5"]  = c.rolling(5).mean()
    df["sma_10"] = c.rolling(10).mean()
    df["sma_20"] = c.rolling(20).mean()
    df["sma_50"] = c.rolling(50).mean()
    df["ema_12"] = c.ewm(span=12, adjust=False).mean()
    df["ema_26"] = c.ewm(span=26, adjust=False).mean()

    # ── MACD ─────────────────────────────────────────────────────────────────
    df["macd"]        = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── RSI (14-period) ───────────────────────────────────────────────────────
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_mean       = c.rolling(20).mean()
    bb_std        = c.rolling(20).std()
    df["bb_upper"] = bb_mean + 2 * bb_std
    df["bb_lower"] = bb_mean - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (bb_mean + 1e-9)
    df["bb_pct"]   = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # ── Volume Features ───────────────────────────────────────────────────────
    df["volume_sma20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / (df["volume_sma20"] + 1e-9)

    # ── Price Distance from Moving Averages ───────────────────────────────────
    df["dist_sma20"] = (c - df["sma_20"]) / (df["sma_20"] + 1e-9)
    df["dist_sma50"] = (c - df["sma_50"]) / (df["sma_50"] + 1e-9)

    # ── Stochastic Oscillator (14-period) ─────────────────────────────────────
    low14         = df["Low"].rolling(14).min()
    high14        = df["High"].rolling(14).max()
    df["stoch_k"] = 100 * (c - low14) / (high14 - low14 + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ── Average True Range (ATR) ──────────────────────────────────────────────
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - c.shift()).abs(),
        (df["Low"]  - c.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr14"] / (c + 1e-9)   # normalized ATR

    # ── On-Balance Volume (OBV) ───────────────────────────────────────────────
    obv = (np.sign(c.diff()) * df["Volume"]).fillna(0).cumsum()
    df["obv_change"] = obv.pct_change().fillna(0)

    # ── Price relative to recent high/low ─────────────────────────────────────
    df["pct_from_52w_high"] = (c / c.rolling(252).max()) - 1
    df["pct_from_52w_low"]  = (c / c.rolling(252).min()) - 1

    # ── Target: 1 if next-day close > today's close, else 0 ──────────────────
    df["target"] = (c.shift(-1) > c).astype(int)

    df.dropna(inplace=True)
    return df


# Canonical list of feature columns used as model input
FEATURE_COLS = [
    "daily_return", "log_return",
    "sma_5", "sma_10", "sma_20", "sma_50",
    "ema_12", "ema_26",
    "macd", "macd_signal", "macd_hist",
    "rsi",
    "bb_width", "bb_pct",
    "volume_ratio",
    "dist_sma20", "dist_sma50",
    "stoch_k", "stoch_d",
    "atr14", "atr_pct",
    "obv_change",
    "pct_from_52w_high", "pct_from_52w_low",
]
