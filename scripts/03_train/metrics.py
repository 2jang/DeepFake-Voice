"""Evaluation metrics for anti-spoofing: EER and accuracy."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Equal Error Rate — the threshold where FAR == FRR."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer)


def compute_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    return float((labels == preds).mean())
