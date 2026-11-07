"""Shared policy for final predictions in physical units."""
import numpy as np


RISK_POLICY = "risk-probability-clip-0-1-v1"


def clip_risk_probabilities(values):
    """Clip finite NumPy arrays or Torch tensors; never hide failed sampling."""
    if isinstance(values, np.ndarray):
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite risk prediction before clipping")
        return np.clip(values, 0.0, 1.0)
    if not bool(values.isfinite().all()):
        raise FloatingPointError("Nonfinite risk prediction before clipping")
    return values.clamp(0.0, 1.0)


def prediction_policy(dataset: str) -> str:
    return RISK_POLICY if dataset == "risk_quantification" else "none"


def postprocess_predictions(dataset: str, fields: dict) -> dict:
    if dataset != "risk_quantification":
        return fields
    return {name: clip_risk_probabilities(values) if name == "risk" else values
            for name, values in fields.items()}
