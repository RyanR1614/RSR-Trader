"""
RSR — News Data Acquisition
Fetches news articles from NewsAPI (free tier) and Yahoo Finance RSS.
No key needed for Yahoo Finance RSS fallback.
"""
import json
import logging
import os
import time
from datetime import datetime, timedelta

import requests

from config.settings import NEWSAPI_KEY, DATA_DIR

logger = logging.getLogger("rsr")


# ── NewsAPI ───────────────────────────────────────────────────────────────────

def fetch_newsapi(query: str, days_back: int = 2) -> list[dict]:
    """
    Fetch articles from NewsAPI free tier.
    Limit: 100 requests/day. Only last 30 days available.
    """
    if not NEWSAPI_KEY:
        logger.warning("NEWSAPI_KEY not set — skipping NewsAPI fetch.")
        return []

    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        query,
        "from":     from_date,
        "sortBy":   "publishedAt",
        "language": "en",
        "apiKey":   NEWSAPI_KEY,
        "pageSize": 50,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
    except Exception as e:
        logger.error(f"NewsAPI request failed: {e}")
        return []

    if data.get("status") != "ok":
        logger.warning(f"NewsAPI error: {data.get('message')}")
        return []

    articles = data.get("articles", [])
    return [
        {
            "title":  (a.get("title")       or "").strip(),
            "body":   (a.get("description") or "").strip(),
            "source": (a.get("source", {}) or {}).get("name", "newsapi"),
            "date":   (a.get("publishedAt") or "")[:10],
        }
        for a in articles
        if a.get("title")
    ]


# ── Yahoo Finance RSS (no key needed) ────────────────────────────────────────

def fetch_yahoo_rss(ticker: str) -> list[dict]:
    """
    Fetch news from Yahoo Finance RSS feed. No API key needed.
    Returns up to ~20 recent headlines.
    """
    try:
        import xml.etree.ElementTree as ET
        url  = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "RSR/1.0"})
        root = ET.fromstring(resp.text)
        items = root.findall(".//item")
        today = datetime.utcnow().strftime("%Y-%m-%d")
        articles = []
        for item in items:
            title = (item.findtext("title") or "").strip()
            desc  = (item.findtext("description") or "").strip()
            pub   = item.findtext("pubDate") or ""
            # Parse date from RFC 822 format
            try:
                dt   = datetime.strptime(pub[:16].strip(), "%a, %d %b %Y")
                date = dt.strftime("%Y-%m-%d")
            except Exception:
                date = today
            if title:
                articles.append({
                    "title":  title,
                    "body":   desc,
                    "source": "yahoo_rss",
                    "date":   date,
                })
        return articles
    except Exception as e:
        logger.warning(f"Yahoo RSS fetch failed for {ticker}: {e}")
        return []


# ── Reddit ────────────────────────────────────────────────────────────────────

def fetch_reddit_posts(ticker: str, limit: int = 40) -> list[dict]:
    """
    Fetch Reddit posts mentioning the ticker via PRAW.
    Searches r/wallstreetbets, r/stocks, r/investing.
    """
    from config.settings import REDDIT_CLIENT_ID, REDDIT_SECRET, REDDIT_USER_AGENT
    if not REDDIT_CLIENT_ID:
        return []
    try:
        import praw
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )
        subreddits = ["wallstreetbets", "stocks", "investing", "StockMarket"]
        posts = []
        per_sub = max(1, limit // len(subreddits))
        for sub in subreddits:
            for post in reddit.subreddit(sub).search(ticker, sort="new", limit=per_sub):
                posts.append({
                    "title":  post.title,
                    "body":   (post.selftext or "")[:400],
                    "source": f"reddit/{sub}",
                    "date":   datetime.utcfromtimestamp(post.created_utc).strftime("%Y-%m-%d"),
                })
            time.sleep(0.5)   # be polite to Reddit's API
        return posts
    except Exception as e:
        logger.warning(f"Reddit fetch failed for {ticker}: {e}")
        return []


# ── Storage ───────────────────────────────────────────────────────────────────

def save_news(ticker: str, articles: list[dict]):
    """Append fetched articles to a per-ticker JSON file, deduplicating by title."""
    os.makedirs(DATA_DIR + "sentiment", exist_ok=True)
    path = f"{DATA_DIR}sentiment/{ticker}_news.json"

    existing = []
    if os.path.exists(path):
        with open(path) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

    combined = existing + articles
    seen, deduped = set(), []
    for a in combined:
        key = a.get("title", "")
        if key and key not in seen:
            seen.add(key)
            deduped.append(a)

    with open(path, "w") as f:
        json.dump(deduped, f, indent=2)
    logger.info(f"  Saved {len(deduped)} total articles for {ticker} → {path}")


def fetch_all_news(ticker: str):
    """Fetch from all sources and save. Called during daily pipeline."""
    logger.info(f"  Fetching news for {ticker}...")
    articles = []
    articles += fetch_newsapi(f"{ticker} stock earnings", days_back=2)
    articles += fetch_yahoo_rss(ticker)
    articles += fetch_reddit_posts(ticker, limit=40)
    save_news(ticker, articles)
    return articles
