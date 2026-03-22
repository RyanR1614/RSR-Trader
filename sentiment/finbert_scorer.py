"""
RSR — Optional FinBERT Scorer
Higher accuracy than VADER. Requires ~400MB download and ideally a GPU.
Usage: replace score_article() calls with finbert_score() if you have GPU resources.
"""
import logging

logger = logging.getLogger("rsr")

_PIPELINE = None


def get_finbert():
    """Lazy-load FinBERT pipeline (downloads ~400MB on first call)."""
    global _PIPELINE
    if _PIPELINE is None:
        try:
            import torch
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Loading FinBERT on {'GPU' if device == 0 else 'CPU'}...")
            _PIPELINE = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                device=device,
                top_k=None,
            )
            logger.info("FinBERT loaded.")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            return None
    return _PIPELINE


def finbert_score(text: str) -> dict:
    """
    Score text with FinBERT.
    Returns: {"positive": float, "negative": float, "neutral": float, "compound": float}
    compound = positive - negative  (in [-1, +1])
    Falls back to neutral on error.
    """
    pipe = get_finbert()
    if pipe is None:
        return {"positive": 0.33, "negative": 0.33, "neutral": 0.34, "compound": 0.0}
    try:
        result = pipe(text[:512], truncation=True)[0]
        scores = {r["label"]: r["score"] for r in result}
        scores["compound"] = scores.get("positive", 0) - scores.get("negative", 0)
        # Normalize keys to match VADER style
        return {
            "pos":      scores.get("positive", 0.33),
            "neg":      scores.get("negative", 0.33),
            "neu":      scores.get("neutral",  0.34),
            "compound": scores["compound"],
        }
    except Exception as e:
        logger.warning(f"FinBERT scoring failed: {e}")
        return {"pos": 0.33, "neg": 0.33, "neu": 0.34, "compound": 0.0}
