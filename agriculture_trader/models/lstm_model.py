"""
LSTM price model.

Architecture: stacked LSTM → dropout → fully-connected → scalar return prediction.
Trains per-ticker; weights saved to disk for inference.
Easily extended to a Transformer by swapping the encoder block.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config.settings import USE_GPU

def _get_device() -> str:
    if USE_GPU:
        if torch.cuda.is_available():
            log.info("GPU: CUDA detected — training on GPU.")
            return "cuda"
        elif torch.backends.mps.is_available():
            log.info("GPU: Apple MPS detected — training on GPU.")
            return "mps"
        else:
            log.warning("USE_GPU=True but no GPU found — falling back to CPU.")
    return "cpu"

from config.settings import (
    LSTM_BATCH_SIZE, LSTM_DROPOUT, LSTM_EPOCHS,
    LSTM_HIDDEN_SIZE, LSTM_LR, LSTM_NUM_LAYERS,
    LSTM_PATIENCE, MODEL_DIR,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────

class LSTMPriceModel(nn.Module):
    """
    Stacked LSTM that maps a (batch, seq_len, n_features) tensor
    to a (batch,) scalar representing the predicted next-bar return.

    To switch to a Transformer encoder, replace the LSTM block with
    nn.TransformerEncoder; the rest of the class is unchanged.
    """

    def __init__(
        self,
        n_features:  int,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers:  int = LSTM_NUM_LAYERS,
        dropout:     float = LSTM_DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        out, _ = self.lstm(x)
        last    = out[:, -1, :]          # take final time step
        last    = self.dropout(last)
        return self.head(last).squeeze(-1)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(
    ticker:    str,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_features: int,
    epochs:      int   = LSTM_EPOCHS,
    batch_size:  int   = LSTM_BATCH_SIZE,
    lr:          float = LSTM_LR,
    patience:    int   = LSTM_PATIENCE,
    device:      Optional[str] = None,
) -> LSTMPriceModel:
    """
    Train the LSTM and return the best model (by validation loss).
    Saves weights to MODEL_DIR/<ticker>.pt.
    """
    device = device or (_get_device())
    log.info(f"[{ticker}] Training on {device}  —  {len(X_tr)} train / {len(X_val)} val samples.")

    model = LSTMPriceModel(n_features).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=3, factor=0.5
    )

    # DataLoaders
    def to_loader(X, y, shuffle):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    tr_loader  = to_loader(X_tr, y_tr, shuffle=True)
    val_loader = to_loader(X_val, y_val, shuffle=False)

    best_val_loss = float("inf")
    stale_epochs  = 0
    best_state    = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(X_tr)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(X_val)

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            log.info(f"[{ticker}] epoch {epoch:3d} | train {tr_loss:.6f} | val {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs  = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                log.info(f"[{ticker}] Early stop at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    _save(model, ticker)
    return model


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

def predict(
    model:   LSTMPriceModel,
    window:  np.ndarray,         # shape (seq_len, n_features) — one window
    device:  Optional[str] = None,
) -> float:
    """
    Predict the scaled next-bar return for a single input window.
    Returns a float in the scaler's feature space ([-1, 1]).
    """
    device = device or (_get_device())
    model.eval()
    x = torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
    with torch.no_grad():
        return model(x).item()


# ─────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────

def _save(model: LSTMPriceModel, ticker: str):
    Path(MODEL_DIR).mkdir(exist_ok=True)
    path = Path(MODEL_DIR) / f"{ticker}.pt"
    torch.save(model.state_dict(), path)
    log.info(f"[{ticker}] Model saved → {path}")


def load_model(ticker: str, n_features: int, device: Optional[str] = None) -> Optional[LSTMPriceModel]:
    path = Path(MODEL_DIR) / f"{ticker}.pt"
    if not path.exists():
        log.warning(f"[{ticker}] No saved model at {path}.")
        return None
    device = device or (_get_device())
    model  = LSTMPriceModel(n_features).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    log.info(f"[{ticker}] Model loaded from {path}")
    return model
