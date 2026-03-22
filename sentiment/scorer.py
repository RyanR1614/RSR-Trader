"""
RSR — Sentiment Scoring
Uses VADER (fast, no GPU) with a finance-specific lexicon extension.
Optional: swap score_article() for finbert_scorer.finbert_score() for higher accuracy.
"""
import json
import logging
import os

import nltk
import pandas as pd

from sentiment.preprocessor import clean_text, combine_article_text
from config.settings import DATA_DIR

logger = logging.getLogger("rsr")

# Download VADER lexicon
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

_SIA = SentimentIntensityAnalyzer()

# Augment VADER with finance-domain vocabulary
_SIA.lexicon.update({
    # Bullish terms
    "bullish":   3.0,  "upside":    2.0,  "soar":      2.5,
    "rally":     2.0,  "surge":     2.5,  "breakout":  2.0,
    "beat":      1.5,  "upgrade":   2.0,  "outperform":2.0,
    "record":    1.5,  "growth":    1.5,  "profit":    1.5,
    "dividend":  1.0,  "buyback":   1.5,  "strong":    1.5,
    # Bearish terms
    "bearish":  -3.0,  "downside": -2.0,  "plunge":   -2.5,
    "crash":    -3.0,  "tumble":   -2.0,  "slump":    -2.0,
    "miss":     -1.5,  "downgrade":-2.0,  "underperform":-2.0,
    "loss":     -1.5,  "debt":     -1.0,  "lawsuit":  -1.5,
    "layoff":   -2.0,  "bankrupt": -3.5,  "fraud":    -3.0,
    "sell":     -1.0,  "weak":     -1.5,  "decline":  -1.5,
})


def score_article(text: str) -> dict:
    """
    Score a piece of text.
    Returns {"pos": float, "neg": float, "neu": float, "compound": float}
    compound is in [-1, +1]; > 0.05 = positive, < -0.05 = negative.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return {"pos": 0.33, "neg": 0.33, "neu": 0.34, "compound": 0.0}
    return _SIA.polarity_scores(cleaned)


def aggregate_daily_sentiment(ticker: str) -> pd.DataFrame:
    """
    Load saved news JSON for a ticker, score each article,
    and return a DataFrame with daily aggregated sentiment scores.
    Index: date (DatetimeIndex)
    Columns: sentiment_pos, sentiment_neg, sentiment_compound, news_count
    """
    path = f"{DATA_DIR}sentiment/{ticker}_news.json"
    if not os.path.exists(path):
        logger.debug(f"No sentiment data for {ticker}")
        return pd.DataFrame()

    with open(path) as f:
        try:
            articles = json.load(f)
        except json.JSONDecodeError:
            return pd.DataFrame()

    if not articles:
        return pd.DataFrame()

    rows = []
    for a in articles:
        text   = combine_article_text(a)
        scores = score_article(text)
        rows.append({
            "date":     a.get("date", ""),
            "pos":      scores["pos"],
            "neg":      scores["neg"],
            "compound": scores["compound"],
            "count":    1,
        })

    df = pd.DataFrame(rows)
    df = df[df["date"] != ""]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    daily = df.groupby("date").agg(
        sentiment_pos      = ("pos",      "mean"),
        sentiment_neg      = ("neg",      "mean"),
        sentiment_compound = ("compound", "mean"),
        news_count         = ("count",    "sum"),
    )
    return daily
