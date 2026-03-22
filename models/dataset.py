"""
RSR — PyTorch Dataset
Builds sliding-window sequences from feature DataFrames.
Scaler is fit once on training data and reused for inference.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class StockDataset(Dataset):
    """
    Sliding-window dataset for sequential stock features.
    Each sample: (lookback_days × num_features) → label (0 or 1)
    """

    def __init__(
        self,
        df,
        feature_cols: list[str],
        lookback: int = 30,
        scaler: StandardScaler = None,
        fit_scaler: bool = True,
    ):
        """
        df:          pandas DataFrame with feature columns + 'target' column
        feature_cols: list of column names to use as features
        lookback:    number of past days per sample
        scaler:      pass an existing fitted scaler for inference; None = fit new
        fit_scaler:  if True, fit scaler on this data (use False for val/test splits)
        """
        self.lookback      = lookback
        self.feature_cols  = feature_cols

        X_raw = df[feature_cols].values.astype(np.float32)
        y_raw = df["target"].values.astype(np.float32)

        # Scale features
        if scaler is not None:
            self.scaler = scaler
            X_scaled = self.scaler.transform(X_raw)
        elif fit_scaler:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_raw)
        else:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_raw)

        # Build sliding windows
        self.X, self.y = [], []
        for i in range(lookback, len(X_scaled)):
            self.X.append(X_scaled[i - lookback : i])
            self.y.append(y_raw[i])

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])

    def get_last_window(self) -> torch.Tensor:
        """Return the most recent window as a (1, lookback, features) tensor for inference."""
        return torch.from_numpy(self.X[-1]).unsqueeze(0)
