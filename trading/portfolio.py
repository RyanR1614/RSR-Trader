"""
RSR — Virtual Portfolio
Tracks cash, share holdings, and full portfolio value history.
Persists state to a JSON file so it survives between runs.
"""
import json
import logging
import os
from datetime import date

from config.settings import STARTING_CASH, PORTFOLIO_FILE

logger = logging.getLogger("rsr")


class Portfolio:
    """
    Thread-safe (single-process) virtual portfolio.

    Attributes:
        cash:      current cash balance
        holdings:  {ticker: num_shares}
        history:   list of daily snapshot dicts
        trades:    list of all trade records
    """

    def __init__(self, starting_cash: float = None, portfolio_file: str = None):
        self.starting_cash  = starting_cash or STARTING_CASH
        self.portfolio_file = portfolio_file or PORTFOLIO_FILE
        self.cash           = self.starting_cash
        self.holdings: dict[str, int]  = {}
        self.history:  list[dict]      = []
        self.trades:   list[dict]      = []
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self):
        if os.path.exists(self.portfolio_file):
            try:
                with open(self.portfolio_file) as f:
                    data = json.load(f)
                self.cash     = data.get("cash",     self.starting_cash)
                self.holdings = data.get("holdings", {})
                self.history  = data.get("history",  [])
                self.trades   = data.get("trades",   [])
                logger.info(
                    f"Portfolio loaded: cash=${self.cash:,.2f} | "
                    f"holdings={list(self.holdings.keys())}"
                )
            except Exception as e:
                logger.warning(f"Could not load portfolio file: {e}. Starting fresh.")

    def save(self):
        os.makedirs(os.path.dirname(self.portfolio_file), exist_ok=True)
        with open(self.portfolio_file, "w") as f:
            json.dump(
                {
                    "cash":     self.cash,
                    "holdings": self.holdings,
                    "history":  self.history,
                    "trades":   self.trades,
                },
                f,
                indent=2,
            )

    def reset(self):
        """Reset portfolio to starting state (use for backtests)."""
        self.cash     = self.starting_cash
        self.holdings = {}
        self.history  = []
        self.trades   = []

    # ── Trading ───────────────────────────────────────────────────────────────

    def buy(self, ticker: str, price: float, num_shares: int) -> bool:
        cost = price * num_shares
        if cost > self.cash:
            logger.warning(
                f"BUY {ticker}: insufficient cash "
                f"(need ${cost:,.2f}, have ${self.cash:,.2f})"
            )
            return False
        if num_shares <= 0:
            return False
        self.cash -= cost
        self.holdings[ticker] = self.holdings.get(ticker, 0) + num_shares
        trade = {
            "date":   str(date.today()),
            "ticker": ticker,
            "action": "BUY",
            "shares": num_shares,
            "price":  round(price, 4),
            "value":  round(cost, 2),
        }
        self.trades.append(trade)
        logger.info(
            f"BUY  {num_shares:6} {ticker:6} @ ${price:8.2f}  "
            f"cost=${cost:10,.2f}  cash_remaining=${self.cash:,.2f}"
        )
        return True

    def sell(self, ticker: str, price: float, num_shares: int = None) -> bool:
        held = self.holdings.get(ticker, 0)
        if held == 0:
            logger.warning(f"SELL {ticker}: no shares held")
            return False
        shares_to_sell = num_shares if num_shares else held
        shares_to_sell = min(shares_to_sell, held)
        proceeds = price * shares_to_sell
        self.cash += proceeds
        self.holdings[ticker] = held - shares_to_sell
        if self.holdings[ticker] == 0:
            del self.holdings[ticker]
        trade = {
            "date":   str(date.today()),
            "ticker": ticker,
            "action": "SELL",
            "shares": shares_to_sell,
            "price":  round(price, 4),
            "value":  round(proceeds, 2),
        }
        self.trades.append(trade)
        logger.info(
            f"SELL {shares_to_sell:6} {ticker:6} @ ${price:8.2f}  "
            f"proceeds=${proceeds:10,.2f}  cash=${self.cash:,.2f}"
        )
        return True

    # ── Valuation ─────────────────────────────────────────────────────────────

    def holdings_value(self, prices: dict[str, float]) -> float:
        return sum(prices.get(t, 0) * s for t, s in self.holdings.items())

    def total_value(self, prices: dict[str, float]) -> float:
        return self.cash + self.holdings_value(prices)

    def record_snapshot(self, prices: dict[str, float]):
        hv    = self.holdings_value(prices)
        total = self.cash + hv
        snap  = {
            "date":           str(date.today()),
            "cash":           round(self.cash, 2),
            "holdings_value": round(hv, 2),
            "total":          round(total, 2),
            "return_pct":     round((total / self.starting_cash - 1) * 100, 4),
        }
        self.history.append(snap)
        logger.info(
            f"Portfolio snapshot: total=${total:,.2f}  "
            f"(cash=${self.cash:,.2f} + stocks=${hv:,.2f})  "
            f"return={snap['return_pct']:+.2f}%"
        )
        return snap

    def summary(self, prices: dict[str, float]) -> str:
        total    = self.total_value(prices)
        ret_pct  = (total / self.starting_cash - 1) * 100
        lines    = [
            "=" * 50,
            f"  RSR Portfolio Summary — {date.today()}",
            "=" * 50,
            f"  Cash:          ${self.cash:>12,.2f}",
            f"  Holdings value:${self.holdings_value(prices):>12,.2f}",
            f"  Total value:   ${total:>12,.2f}",
            f"  Total return:  {ret_pct:>+11.2f}%",
            f"  Positions:     {len(self.holdings)}",
        ]
        for ticker, shares in self.holdings.items():
            price = prices.get(ticker, 0)
            lines.append(f"    {ticker:6}: {shares:5} shares @ ${price:.2f} = ${shares*price:,.2f}")
        lines.append("=" * 50)
        return "\n".join(lines)
