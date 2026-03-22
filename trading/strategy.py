"""
RSR — Trading Strategy
Combines model signal (prob_up) and sentiment to decide BUY / SELL / HOLD.
All thresholds and weights are configurable.
"""
import logging
from config.settings import MAX_POSITION_PCT, TRADE_SIZE_PCT

logger = logging.getLogger("rsr")

# ── Strategy parameters (tune these) ─────────────────────────────────────────
MODEL_WEIGHT     = 0.70    # Weight given to model's P(up) signal
SENTIMENT_WEIGHT = 0.30    # Weight given to sentiment score

BUY_THRESHOLD    = 0.60    # composite signal must exceed this to BUY
SELL_THRESHOLD   = 0.40    # composite signal must be below this to SELL

# Stop-loss: sell if position is down this % from average cost
STOP_LOSS_PCT    = 0.08    # 8% stop-loss


def composite_signal(prob_up: float, sentiment_compound: float) -> float:
    """
    Combine model probability and sentiment into a single [0, 1] signal.
    prob_up:             model output in [0, 1]
    sentiment_compound:  VADER/FinBERT compound in [-1, +1]
    """
    # Normalize sentiment from [-1,1] to [0,1]
    sentiment_norm = (sentiment_compound + 1.0) / 2.0
    return MODEL_WEIGHT * prob_up + SENTIMENT_WEIGHT * sentiment_norm


def make_decision(
    ticker:     str,
    price:      float,
    prob_up:    float,
    sentiment:  float,
    portfolio,
    total_value: float,
    avg_cost:   float = None,
) -> str:
    """
    Decide BUY / SELL / HOLD for one ticker.

    Returns the action string.
    """
    signal      = composite_signal(prob_up, sentiment)
    shares_held = portfolio.holdings.get(ticker, 0)

    logger.debug(
        f"{ticker}: prob_up={prob_up:.3f} sentiment={sentiment:.3f} "
        f"signal={signal:.3f} held={shares_held}"
    )

    # ── Stop-loss check ───────────────────────────────────────────────────────
    if shares_held > 0 and avg_cost and price < avg_cost * (1 - STOP_LOSS_PCT):
        logger.info(f"  STOP-LOSS triggered for {ticker} (cost=${avg_cost:.2f} price=${price:.2f})")
        portfolio.sell(ticker, price)
        return "STOP_LOSS"

    # ── BUY ───────────────────────────────────────────────────────────────────
    if signal > BUY_THRESHOLD:
        current_exposure = (shares_held * price) / (total_value + 1e-9)
        if current_exposure < MAX_POSITION_PCT:
            budget     = total_value * TRADE_SIZE_PCT
            num_shares = int(budget / price)
            if num_shares > 0 and budget <= portfolio.cash:
                portfolio.buy(ticker, price, num_shares)
                return "BUY"
            else:
                logger.debug(f"  BUY {ticker}: not enough cash (budget=${budget:.2f})")

    # ── SELL ──────────────────────────────────────────────────────────────────
    elif signal < SELL_THRESHOLD and shares_held > 0:
        portfolio.sell(ticker, price)
        return "SELL"

    return "HOLD"
