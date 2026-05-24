"""Evaluation metrics for anti-spoofing."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Equal Error Rate — the threshold where FAR == FRR."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer)


def compute_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    return float((labels == preds).mean())


def compute_all(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    preds = (scores >= 0.5).astype(int)
    return {
        "eer": compute_eer(labels, scores),
        "accuracy": compute_accuracy(labels, preds),
        "f1": float(f1_score(labels, preds)),
        "precision": float(precision_score(labels, preds)),
        "recall": float(recall_score(labels, preds)),
        "auc": float(roc_auc_score(labels, scores)),
    }
