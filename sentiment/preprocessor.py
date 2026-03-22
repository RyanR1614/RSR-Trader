"""
RSR — Text Preprocessing for Sentiment Analysis
Cleans and normalizes raw news/social text before scoring.
"""
import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger("rsr")

# Download NLTK data if not already present
for resource in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Finance-specific stopwords to remove (generic financial boilerplate)
FINANCE_NOISE = {
    "said", "says", "company", "shares", "stock", "market",
    "percent", "year", "quarter", "report", "reported", "reuters",
    "bloomberg", "cnbc", "press", "release",
}
STOP_WORDS.update(FINANCE_NOISE)


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline:
      1. Lowercase
      2. Remove URLs
      3. Remove non-alphabetic characters
      4. Tokenize
      5. Remove stopwords
      6. Lemmatize
    Returns cleaned, space-joined string.
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # strip URLs
    text = re.sub(r"[^a-z\s]", " ", text)                 # keep only letters
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def combine_article_text(article: dict) -> str:
    """Combine title and body into one string for scoring."""
    title = article.get("title", "") or ""
    body  = article.get("body", "")  or ""
    return (title + " " + body).strip()
