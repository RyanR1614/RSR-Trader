"""
RSR — Reinforcement Learning Training
Trains a PPO agent using Stable Baselines3.
The agent learns from the RSRTradingEnv custom environment.

Usage:
    python pipeline/train_rl.py              # train on all tickers
    python pipeline/train_rl.py AAPL         # train on one ticker
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TICKERS, DATA_DIR, MODEL_DIR, LOG_FILE

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger("rsr")


def train_rl(ticker: str, timesteps: int = 200_000):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
    except ImportError:
        logger.error("stable-baselines3 not installed. Run: pip install stable-baselines3 gymnasium")
        return

    import pandas as pd
    from data.fetch_prices import load_prices
    from features.combine import build_full_feature_set, ALL_FEATURE_COLS
    from trading.rl_env import RSRTradingEnv

    logger.info(f"Training RL agent for {ticker}...")

    df_raw = load_prices(ticker)
    df     = build_full_feature_set(ticker, df_raw)

    if len(df) < 100:
        logger.warning(f"Not enough data for {ticker}. Skipping RL training.")
        return

    # Use first 80% for training
    split = int(0.80 * len(df))
    train_df = df.iloc[:split].reset_index(drop=True)

    env = RSRTradingEnv(train_df, ALL_FEATURE_COLS)

    # Validate env (SB3 compatibility check)
    try:
        check_env(env, warn=True)
    except Exception as e:
        logger.warning(f"Env check warning: {e}")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,   # exploration bonus
        tensorboard_log=f"logs/ppo_{ticker}",
    )

    model.learn(total_timesteps=timesteps, progress_bar=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = f"{MODEL_DIR}{ticker}_ppo_agent"
    model.save(save_path)
    logger.info(f"RL agent saved → {save_path}.zip")

    # Quick eval on test data
    test_df  = df.iloc[split:].reset_index(drop=True)
    test_env = RSRTradingEnv(test_df, ALL_FEATURE_COLS)
    obs, _   = test_env.reset()
    done     = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = test_env.step(int(action))

    final_val = info.get("portfolio_value", test_env._portfolio_value(test_env._get_price(test_env.step_idx)))
    ret       = (final_val / test_env.initial_cash - 1) * 100
    logger.info(f"RL backtest on test data: final=${final_val:,.0f}  return={ret:+.1f}%")
    logger.info(f"Total trades: {len(test_env.trade_log)}")


if __name__ == "__main__":
    tickers = sys.argv[1:] if len(sys.argv) > 1 else TICKERS[:1]
    for t in tickers:
        try:
            train_rl(t, timesteps=200_000)
        except FileNotFoundError:
            logger.error(f"No data for {t}. Run: python data/fetch_prices.py")
        except Exception as e:
            logger.error(f"RL training failed for {t}: {e}", exc_info=True)
