"""
RSR — Reinforcement Learning Environment (Optional Advanced Module)
Custom Gymnasium environment for training a PPO trading agent.

State:  technical + sentiment features + position info (shares held, cash ratio)
Actions: 0=HOLD, 1=BUY 10% of portfolio, 2=SELL all
Reward:  daily portfolio return %

Usage:
    python trading/rl_env.py          # quick training demo
    python pipeline/train_rl.py       # full RL training
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("rsr")

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    logger.warning("gymnasium not installed. RL environment unavailable. Run: pip install gymnasium")


if GYM_AVAILABLE:
    class RSRTradingEnv(gym.Env):
        """
        Custom single-asset trading environment for RSR.
        Compatible with Stable Baselines3.
        """
        metadata = {"render_modes": ["human"]}

        def __init__(
            self,
            df:            pd.DataFrame,
            feature_cols:  list[str],
            initial_cash:  float = 100_000.0,
            transaction_cost: float = 0.001,   # 0.1% per trade
        ):
            super().__init__()

            self.df               = df.reset_index(drop=True)
            self.feature_cols     = feature_cols
            self.initial_cash     = initial_cash
            self.transaction_cost = transaction_cost
            self.n_features       = len(feature_cols)

            # ── Action space: 0=HOLD, 1=BUY, 2=SELL ─────────────────────────
            self.action_space = spaces.Discrete(3)

            # ── Observation space: features + [shares_norm, cash_ratio, returns_norm] ──
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.n_features + 3,),
                dtype=np.float32,
            )

            # Pre-scale features once for efficiency
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
            feature_matrix = self.df[self.feature_cols].values.astype(np.float32)
            self._features_scaled = self._scaler.fit_transform(feature_matrix)

            self._reset_state()

        def _reset_state(self):
            self.step_idx   = 30
            self.cash       = float(self.initial_cash)
            self.shares     = 0.0
            self.prev_value = self.initial_cash
            self.buy_price  = 0.0
            self.trade_log  = []

        def _get_price(self, idx: int) -> float:
            return float(self.df.iloc[idx]["Close"])

        def _portfolio_value(self, price: float) -> float:
            return self.cash + self.shares * price

        def _get_obs(self) -> np.ndarray:
            features = self._features_scaled[self.step_idx]
            price    = self._get_price(self.step_idx)
            port_val = self._portfolio_value(price)

            shares_norm = np.clip(self.shares * price / (port_val + 1e-9), 0, 1)
            cash_ratio  = np.clip(self.cash / (port_val + 1e-9), 0, 1)
            ret_norm    = np.clip((port_val / self.initial_cash - 1), -1, 1)

            return np.concatenate([features, [shares_norm, cash_ratio, ret_norm]]).astype(np.float32)

        def step(self, action: int):
            price = self._get_price(self.step_idx)

            # ── Execute action ────────────────────────────────────────────────
            if action == 1:   # BUY: invest 10% of portfolio
                budget = self._portfolio_value(price) * 0.10
                if budget > self.cash and self.shares == 0:
                    n = int(self.cash * 0.10 / price)
                else:
                    n = int(budget / price)
                if n > 0 and self.cash >= n * price * (1 + self.transaction_cost):
                    cost = n * price * (1 + self.transaction_cost)
                    self.cash    -= cost
                    self.shares  += n
                    self.buy_price = price
                    self.trade_log.append(("BUY", self.step_idx, price, n))

            elif action == 2 and self.shares > 0:   # SELL all
                proceeds = self.shares * price * (1 - self.transaction_cost)
                self.cash   += proceeds
                self.shares  = 0
                self.trade_log.append(("SELL", self.step_idx, price, 0))

            # ── Step forward ──────────────────────────────────────────────────
            self.step_idx += 1
            done = self.step_idx >= len(self.df) - 1

            next_price = self._get_price(self.step_idx)
            new_value  = self._portfolio_value(next_price)

            # ── Reward: daily return + small profit/loss bonus ────────────────
            daily_return = (new_value - self.prev_value) / (self.prev_value + 1e-9)
            reward = daily_return * 100   # scale reward

            # Small penalty for holding cash doing nothing (encourages activity)
            if self.shares == 0:
                reward -= 0.001

            self.prev_value = new_value

            return self._get_obs(), reward, done, False, {"portfolio_value": new_value}

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self._reset_state()
            return self._get_obs(), {}

        def render(self):
            price = self._get_price(self.step_idx)
            val   = self._portfolio_value(price)
            ret   = (val / self.initial_cash - 1) * 100
            print(
                f"Step {self.step_idx:4d} | "
                f"Price=${price:.2f} | "
                f"Shares={self.shares:.0f} | "
                f"Cash=${self.cash:,.0f} | "
                f"Total=${val:,.0f} | "
                f"Return={ret:+.2f}%"
            )
