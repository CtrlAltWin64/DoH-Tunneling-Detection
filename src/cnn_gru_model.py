"""1D-CNN + GRU hybrid classifier for adaptive DoH-tunnelling detection.

Architecture:
  Input  : (batch, seq_len, n_features) -- ordered per-packet metadata vectors
  Stage 1: Two Conv1d layers (local pattern extraction across consecutive packets)
  Stage 2: GRU (temporal dependency and evolving behaviour)
  Head   : Linear -> sigmoid (malicious-flow probability)

Design notes:
- Variable-length sequences are handled by zero-padding + packing.
- BatchNorm only on training data (statistics never leak from val/test).
- Class-imbalance handled via pos_weight in BCEWithLogitsLoss.
- Deterministic seeding where practical (torch.manual_seed before weight init).
- Validation-based early stopping with configurable patience.
- Checkpoint saving on every validation improvement.
- The model is trained once and scored at every checkpoint t by feeding
  only the first t packets of each flow (causal inference).
"""
from __future__ import annotations

import logging
import math
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Masked BatchNorm for variable-length sequences
# ---------------------------------------------------------------------------

class MaskedBatchNorm1d(nn.Module):
    """1D Batch Normalization that computes statistics over valid timesteps only.

    Prevents padded timesteps from contaminating mean/variance estimates or
    leaking across variable-length sequences in a batch. Padded positions
    are explicitly zeroed after normalization.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, channels, length)
        mask : Tensor, shape (batch, length), bool, True for valid positions.
        """
        if mask is None:
            mask = torch.ones(x.size(0), x.size(2), dtype=torch.bool, device=x.device)
        mask_f = mask.unsqueeze(1).float()  # (batch, 1, length)

        if self.training:
            total_valid = mask_f.sum(dim=(0, 2)).clamp(min=1.0)
            x_valid = x * mask_f
            mean = x_valid.sum(dim=(0, 2)) / total_valid
            var = (((x - mean.view(1, -1, 1)) ** 2) * mask_f).sum(dim=(0, 2)) / total_valid
            with torch.no_grad():
                self.running_mean.mul_(1.0 - self.momentum).add_(mean * self.momentum)
                self.running_var.mul_(1.0 - self.momentum).add_(var * self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean.view(1, -1, 1)) / torch.sqrt(var.view(1, -1, 1) + self.eps)
        x_norm = x_norm * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return x_norm * mask_f


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CnnGruClassifier(nn.Module):
    """1D-CNN + GRU binary classifier with causal convolutions and masked normalization.

    Parameters
    ----------
    n_features : int
        Number of input features per time step (packet).
    conv_channels : int
        Number of output channels for the Conv1d layers.
    conv_kernel : int
        Kernel width for both Conv1d layers (must be <= shortest prefix length).
    gru_hidden : int
        Hidden size of the GRU.
    gru_layers : int
        Number of stacked GRU layers.
    dropout : float
        Dropout rate applied between GRU layers (only if gru_layers > 1).
    """

    def __init__(
        self,
        n_features: int,
        conv_channels: int = 64,
        conv_kernel: int = 3,
        gru_hidden: int = 64,
        gru_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_features = n_features
        self.conv_kernel = conv_kernel

        # --- Causal Conv + Masked BatchNorm block ------------------------
        self.conv1 = nn.Conv1d(n_features, conv_channels, kernel_size=conv_kernel, padding=0)
        self.bn1 = MaskedBatchNorm1d(conv_channels)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=conv_kernel, padding=0)
        self.bn2 = MaskedBatchNorm1d(conv_channels)
        self.relu = nn.ReLU()

        # --- GRU block ---------------------------------------------------
        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # --- Classification head ----------------------------------------
        self.head = nn.Linear(gru_hidden, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal convolution and masked normalization.

        Parameters
        ----------
        x : Tensor, shape (batch, max_seq_len, n_features)
        lengths : Tensor, shape (batch,) -- actual sequence lengths (before padding)

        Returns
        -------
        logits : Tensor, shape (batch,)  -- raw log-odds (positive = malicious)
        """
        # Validate inputs
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input tensor (batch, seq_len, features), got ndim={x.ndim}")
        if lengths.ndim != 1:
            raise ValueError(f"Expected 1D lengths tensor, got ndim={lengths.ndim}")
        if len(x) != len(lengths):
            raise ValueError(f"Batch size mismatch: x has {len(x)} samples but lengths has {len(lengths)}")
        if len(x) == 0:
            raise ValueError("Empty batch is not supported")
        if x.size(-1) != self.n_features:
            raise ValueError(f"Feature dimension mismatch: expected {self.n_features}, got {x.size(-1)}")
        if (lengths <= 0).any() or (lengths > x.size(1)).any():
            raise ValueError(f"Invalid sequence lengths: must be in [1, {x.size(1)}], got min={int(lengths.min())}, max={int(lengths.max())}")

        bs, max_len, _ = x.shape
        device = x.device

        # Create validity mask: shape (B, L)
        steps = torch.arange(max_len, device=device).unsqueeze(0)  # (1, L)
        mask = steps < lengths.to(device).unsqueeze(1)              # (B, L) bool

        # Permute for Conv1d: (batch, channels, seq_len)
        xc = x.permute(0, 2, 1)

        # Stage 1: Causal Conv1d (pad left by conv_kernel - 1, 0 on right)
        xc = torch.nn.functional.pad(xc, (self.conv_kernel - 1, 0))
        xc = self.conv1(xc)
        xc = self.bn1(xc, mask)
        xc = self.relu(xc)

        # Stage 2: Second Causal Conv1d
        xc = torch.nn.functional.pad(xc, (self.conv_kernel - 1, 0))
        xc = self.conv2(xc)
        xc = self.bn2(xc, mask)
        xc = self.relu(xc)

        # Permute back for GRU: (batch, seq_len, conv_channels)
        xc = xc.permute(0, 2, 1)

        # Pack padded sequence to skip padded timesteps in GRU
        packed = pack_padded_sequence(
            xc, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)             # h_n: (num_layers, B, H)
        last_h = h_n[-1]                       # (B, H)
        logits = self.head(last_h).squeeze(-1) # (B,)
        return logits

    def load_state_dict(self, state_dict, strict=True):
        """Supports remapping legacy sequential conv parameters."""
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("conv.0."):
                remapped["conv1." + k[7:]] = v
            elif k.startswith("conv.1."):
                remapped["bn1." + k[7:]] = v
            elif k.startswith("conv.3."):
                remapped["conv2." + k[7:]] = v
            elif k.startswith("conv.4."):
                remapped["bn2." + k[7:]] = v
            else:
                remapped[k] = v
        return super().load_state_dict(remapped, strict=strict)


# ---------------------------------------------------------------------------
# Normalizer (fit on training data only)
# ---------------------------------------------------------------------------

class SequenceNormalizer:
    """Standardises per-feature statistics computed on training sequences only."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "SequenceNormalizer":
        """X shape: (n_samples, seq_len, n_features) or (n_total_timesteps, n_features)."""
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = flat.mean(axis=0, keepdims=True)
        std = flat.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0           # avoid division by zero for constant features
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("SequenceNormalizer not fitted")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def state_dict(self) -> dict:
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_state_dict(cls, d: dict) -> "SequenceNormalizer":
        n = cls()
        n.mean_ = np.asarray(d["mean"], dtype=np.float32)
        n.std_ = np.asarray(d["std"], dtype=np.float32)
        return n


# ---------------------------------------------------------------------------
# Dataset / collate
# ---------------------------------------------------------------------------

def _build_sequence_array(flow, t: int) -> np.ndarray:
    """Return shape (t, n_features) packet-metadata matrix for the prefix of length t."""
    ts = np.asarray(flow.ts[:t], dtype=np.float64)
    ln = np.asarray(flow.length[:t], dtype=np.float64)
    dr = np.asarray(flow.direction[:t], dtype=np.float64)

    fwd = dr > 0
    cum_fwd = np.cumsum(fwd).astype(np.float64)
    cum_bwd = np.cumsum(~fwd).astype(np.float64)
    cum_bytes = np.cumsum(ln)
    cum_fwd_bytes = np.cumsum(np.where(fwd, ln, 0.0))
    cum_bwd_bytes = np.cumsum(np.where(~fwd, ln, 0.0))

    # IAT (0 for first packet)
    iat = np.concatenate([[0.0], np.diff(ts)])
    elapsed = ts - ts[0]

    # Per-packet feature matrix: (t, 8)
    # [pkt_len, direction(+1/-1), iat, elapsed, cum_pkts_fwd, cum_pkts_bwd, cum_bytes_fwd, cum_bytes_bwd]
    mat = np.stack([
        ln,         # packet length
        dr,         # direction (+1 / -1)
        iat,        # inter-arrival time
        elapsed,    # elapsed time from start
        cum_fwd,    # cumulative forward packets
        cum_bwd,    # cumulative backward packets
        cum_fwd_bytes,  # cumulative forward bytes
        cum_bwd_bytes,  # cumulative backward bytes
    ], axis=1).astype(np.float32)   # (t, 8)
    return mat


SEQ_FEATURE_NAMES = [
    "pkt_len", "direction", "iat", "elapsed",
    "cum_pkts_fwd", "cum_pkts_bwd", "cum_bytes_fwd", "cum_bytes_bwd",
]
N_SEQ_FEATURES = len(SEQ_FEATURE_NAMES)


def build_sequence_dataset(flows, t: int):
    """Build sequences and labels for all flows with >= t packets.

    Returns
    -------
    seqs  : list of np.ndarray, each shape (actual_len, N_SEQ_FEATURES) -- length == t here
    labels: np.ndarray, shape (n,)
    flow_ids: list[str]
    """
    seqs, labels, flow_ids = [], [], []
    for fl in flows:
        if len(fl.ts) < t:
            continue
        seqs.append(_build_sequence_array(fl, t))
        labels.append(fl.label)
        flow_ids.append(fl.flow_id)
    return seqs, np.asarray(labels, dtype=np.float32), flow_ids


def collate_sequences(batch):
    """Collate list of (seq_array, label) into padded tensors.

    Returns: (padded_seqs, lengths, labels)
    """
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    padded = pad_sequence(
        [torch.from_numpy(s) for s in seqs], batch_first=True, padding_value=0.0
    )
    return padded, lengths, torch.tensor(labels, dtype=torch.float32)


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, labels, normalizer: Optional[SequenceNormalizer] = None):
        if normalizer is not None:
            self.seqs = [normalizer.transform(s) for s in seqs]
        else:
            self.seqs = list(seqs)
        self.labels = labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx].astype(np.float32), float(self.labels[idx])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _pos_weight(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    n_neg = (labels == 0).sum()
    n_pos = (labels == 1).sum()
    if n_pos == 0 or n_neg == 0:
        return torch.ones(1, device=device)
    return torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)


def train_cnn_gru(
    train_flows,
    val_flows,
    checkpoints,
    cfg,
    checkpoint_path: Path,
    device: Optional[str] = None,
) -> tuple["CnnGruClassifier", SequenceNormalizer, dict]:
    """Train the CNN-GRU on progressive-prefix sequences from training flows.

    Training uses ALL prefixes of ALL training flows at ALL checkpoints (mixed into
    one dataset). A validation loss is computed on val_flows with the same scheme.
    The model achieving the lowest validation loss is saved and returned.

    Parameters
    ----------
    train_flows : list[Flow]
    val_flows   : list[Flow]
    checkpoints : sequence of int  -- e.g. (10, 20, 30, 40)
    cfg         : Config object
    checkpoint_path : Path  -- where to save the best model weights
    device : str or None  -- 'cpu', 'cuda', or None (auto-detect)

    Returns
    -------
    model       : best CnnGruClassifier
    normalizer  : SequenceNormalizer fitted on training data
    train_record: dict with training history
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # --- Build training sequences (all checkpoints) ----------------------
    train_seqs_all, train_labels_all = [], []
    for t in checkpoints:
        seqs, labels, _ = build_sequence_dataset(train_flows, t)
        train_seqs_all.extend(seqs)
        train_labels_all.append(labels)
    if not train_seqs_all:
        raise ValueError("No training sequences could be built (all flows too short?)")
    train_labels_all = np.concatenate(train_labels_all)

    # Fit normalizer on all training timesteps
    all_train_mat = np.vstack(train_seqs_all)   # (total_timesteps, F)
    normalizer = SequenceNormalizer().fit(all_train_mat)

    # Validation sequences (largest checkpoint only for speed, but use all)
    val_seqs_all, val_labels_all = [], []
    for t in checkpoints:
        seqs, labels, _ = build_sequence_dataset(val_flows, t)
        val_seqs_all.extend(seqs)
        val_labels_all.append(labels)
    if not val_seqs_all:
        raise ValueError("No validation sequences could be built")
    val_labels_all = np.concatenate(val_labels_all)

    train_ds = SequenceDataset(train_seqs_all, train_labels_all, normalizer)
    val_ds = SequenceDataset(val_seqs_all, val_labels_all, normalizer)

    bs = getattr(cfg, "cnn_batch_size", 128)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        collate_fn=collate_sequences, drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        collate_fn=collate_sequences,
    )

    # --- Model ----------------------------------------------------------
    model = CnnGruClassifier(
        n_features=N_SEQ_FEATURES,
        conv_channels=getattr(cfg, "cnn_conv_channels", 64),
        conv_kernel=getattr(cfg, "cnn_conv_kernel", 3),
        gru_hidden=getattr(cfg, "cnn_gru_hidden", 64),
        gru_layers=getattr(cfg, "cnn_gru_layers", 2),
        dropout=getattr(cfg, "cnn_dropout", 0.2),
    ).to(dev)

    pos_w = _pos_weight(train_labels_all, dev)
    n_benign = int((train_labels_all == 0).sum())
    n_malicious = int((train_labels_all == 1).sum())

    logger.info(
        "[CNN-GRU] Training sequences: "
        f"benign={n_benign}, malicious={n_malicious}, "
        f"malicious pos_weight={pos_w.item():.6f}"
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=getattr(cfg, "cnn_lr", 1e-3))

    epochs = getattr(cfg, "cnn_epochs", 30)
    patience = getattr(cfg, "cnn_patience", 7)

    best_val_loss = float("inf")
    patience_best_loss = float("inf")
    min_delta = getattr(cfg, "cnn_min_delta", 0.0)
    if epochs < 1 or patience < 1 or min_delta < 0:
        raise ValueError("cnn_epochs and cnn_patience must be positive; cnn_min_delta must be nonnegative")
    no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # --- Train -------------------------------------------------------
        model.train()
        total_loss, total_n = 0.0, 0
        for padded, lengths, labels in train_loader:
            padded = padded.to(dev)
            lengths = lengths.to(dev)
            labels = labels.to(dev)
            optimizer.zero_grad()
            logits = model(padded, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total_n += len(labels)
        train_loss = total_loss / total_n

        # --- Validate ----------------------------------------------------
        model.eval()
        vtotal_loss, vtotal_n = 0.0, 0
        with torch.no_grad():
            for padded, lengths, labels in val_loader:
                padded = padded.to(dev)
                lengths = lengths.to(dev)
                labels = labels.to(dev)
                logits = model(padded, lengths)
                loss = criterion(logits, labels)
                vtotal_loss += loss.item() * len(labels)
                vtotal_n += len(labels)
        val_loss = vtotal_loss / vtotal_n

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, normalizer, checkpoint_path,
                       extra={"epoch": epoch, "val_loss": val_loss})
            logger.debug(f"  epoch {epoch}: val_loss={val_loss:.8g} *** saved")

        # Saving the true minimum and counting material patience improvements
        # are separate: even a sub-min_delta improvement must remain loadable.
        if val_loss < patience_best_loss - min_delta:
            patience_best_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            logger.debug(f"  epoch {epoch}: train={train_loss:.8g} val={val_loss:.8g} (no material improvement {no_improve})")
            if no_improve >= patience:
                logger.info(f"[CNN-GRU] early stopping at epoch {epoch} (patience={patience})")
                break

    # Reload best weights
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    normalizer = SequenceNormalizer.from_state_dict(ckpt["normalizer"])
    history["best_epoch"] = int(ckpt["epoch"])
    history["best_val_loss"] = float(ckpt["val_loss"])
    history["epochs_trained"] = len(history["train_loss"])
    history["patience_min_delta"] = float(min_delta)
    return model, normalizer, history


# ---------------------------------------------------------------------------
# Inference (causal prefix scoring)
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_flows_at_checkpoint(
    flows,
    t: int,
    model: CnnGruClassifier,
    normalizer: SequenceNormalizer,
    device: str = "cpu",
    batch_size: int = 256,
) -> tuple[np.ndarray, list[str]]:
    """Return (probabilities, flow_ids) for flows with >= t packets.

    Only the first t packets of each flow are used (causal).
    Probabilities are sigmoid-transformed logits (NOT calibrated -- calibration
    is applied separately in the pipeline).
    """
    dev = torch.device(device)
    seqs, _, flow_ids = build_sequence_dataset(flows, t)
    if not seqs:
        return np.array([]), []

    seqs_norm = [normalizer.transform(s).astype(np.float32) for s in seqs]
    model.eval()
    model.to(dev)

    probs = []
    for i in range(0, len(seqs_norm), batch_size):
        batch = seqs_norm[i: i + batch_size]
        lengths = torch.tensor([s.shape[0] for s in batch], dtype=torch.long)
        padded = pad_sequence(
            [torch.from_numpy(s) for s in batch], batch_first=True, padding_value=0.0
        ).to(dev)
        logits = model(padded, lengths)
        probs.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(probs), flow_ids


# ---------------------------------------------------------------------------
# Checkpoint I/O helpers
# ---------------------------------------------------------------------------

def save_model(model: CnnGruClassifier, normalizer: SequenceNormalizer,
               path: Path, extra: dict | None = None) -> None:
    payload = {
        "schema_version": 1,
        "model_state": model.state_dict(),
        "normalizer": normalizer.state_dict(),
        "model_kwargs": {
            "n_features": model.n_features,
            "conv_channels": model.conv1.out_channels if hasattr(model, "conv1") else model.conv[0].out_channels,
            "conv_kernel": getattr(model, "conv_kernel", model.conv1.kernel_size[0] if hasattr(model, "conv1") else model.conv[0].kernel_size[0]),
            "gru_hidden": model.gru.hidden_size,
            "gru_layers": model.gru.num_layers,
            "dropout": model.gru.dropout,
        },
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_model(path: Path, device: str = "cpu") -> tuple[CnnGruClassifier, SequenceNormalizer]:
    """Load current or legacy training checkpoints without modifying their files.

    Legacy training artifacts omitted architecture metadata. Tensor dimensions
    recover the architecture; dropout is read from the companion run record
    when available (it is inactive in the returned inference model).
    """
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    kw = payload.get("model_kwargs")
    if kw is None:
        state = payload["model_state"]
        conv = state["conv1.weight"] if "conv1.weight" in state else state["conv.0.weight"]
        layers = sum(k.startswith("gru.weight_ih_l") for k in state)
        dropout = 0.0
        record_path = path.parent / "run_record.json"
        if record_path.exists():
            config = json.loads(record_path.read_text(encoding="utf-8")).get("config", {})
            dropout = float(config.get("cnn_dropout", 0.0))
        kw = {
            "n_features": int(conv.shape[1]),
            "conv_channels": int(conv.shape[0]),
            "conv_kernel": int(conv.shape[2]),
            "gru_hidden": int(state["gru.weight_hh_l0"].shape[1]),
            "gru_layers": layers,
            "dropout": dropout,
        }
    model = CnnGruClassifier(**kw)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    normalizer = SequenceNormalizer.from_state_dict(payload["normalizer"])
    return model, normalizer
