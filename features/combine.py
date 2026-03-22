"""
RSR — Feature Combination
Merges technical indicators with daily sentiment scores into one DataFrame.
"""
import pandas as pd
from features.technical import add_technical_features, FEATURE_COLS
from sentiment.scorer import aggregate_daily_sentiment

SENTIMENT_COLS = [
    "sentiment_pos",
    "sentiment_neg",
    "sentiment_compound",
    "news_count",
]

ALL_FEATURE_COLS = FEATURE_COLS + SENTIMENT_COLS


def build_full_feature_set(ticker: str, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Compute technical indicators from price data.
    2. Load and merge daily sentiment scores.
    3. Return a clean DataFrame with ALL_FEATURE_COLS + target + Close.
    """
    # Step 1: technical indicators
    df = add_technical_features(price_df.copy())

    # Step 2: sentiment
    sentiment = aggregate_daily_sentiment(ticker)

    if not sentiment.empty:
        df = df.join(sentiment, how="left")
        df["sentiment_pos"]      = df["sentiment_pos"].fillna(0.33)
        df["sentiment_neg"]      = df["sentiment_neg"].fillna(0.33)
        df["sentiment_compound"] = df["sentiment_compound"].fillna(0.0)
        df["news_count"]         = df["news_count"].fillna(0)
    else:
        df["sentiment_pos"]      = 0.33
        df["sentiment_neg"]      = 0.33
        df["sentiment_compound"] = 0.0
        df["news_count"]         = 0.0

    keep = ALL_FEATURE_COLS + ["target", "Close"]
    return df[keep].dropna()
