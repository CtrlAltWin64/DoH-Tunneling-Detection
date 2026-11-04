"""Load calibrated CNN inference bundles produced by the experiment pipeline.

Only load trusted local checkpoints: torch serialization can contain Python objects.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .calibration import Calibrator
from .cnn_gru_model import load_model, score_flows_at_checkpoint


@dataclass
class CnnBundle:
    model: object
    normalizer: object
    calibrator: Calibrator
    config: dict
    thresholds: dict
    device: str = "cpu"

    def predict_proba(self, flows, checkpoint: int):
        if checkpoint not in self.config["checkpoints"]:
            raise ValueError("Checkpoint was not included in this experiment")
        raw, ids = score_flows_at_checkpoint(
            flows, checkpoint, self.model, self.normalizer, device=self.device
        )
        return self.calibrator.transform(raw, np.full(len(raw), checkpoint)), ids


def load_cnn_bundle(path: str | Path, device: str = "cpu") -> CnnBundle:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format_version") != 1 or "calibrator" not in payload:
        raise ValueError("Expected a version 1 calibrated CNN bundle")
    model, normalizer = load_model(Path(path), device=device)
    return CnnBundle(model, normalizer, Calibrator.from_description(payload["calibrator"]),
                     payload["config"], payload["thresholds"], device)
