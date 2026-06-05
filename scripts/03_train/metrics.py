"""Evaluation metrics for anti-spoofing.

지표:
  - EER: Equal Error Rate (FAR=FRR 교차점)
  - minDCF: ASVspoof 표준 비용 함수 (C_miss=2, C_fa=1, P_target=0.5)
  - FPR: real 음성을 fake로 오판하는 비율 (임동훈 2024 Table 5 비교용)
  - FNR: fake 음성을 real로 오판하는 비율
  - Accuracy, F1, Precision, Recall, AUC: 표준 분류 지표 (threshold=0.5)
"""

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


def compute_mindcf(
    labels: np.ndarray,
    scores: np.ndarray,
    c_miss: float = 2.0,
    c_fa: float = 1.0,
    p_target: float = 0.5,
) -> float:
    """Normalized minimum Detection Cost Function (ASVspoof 표준).

    임동훈(2024) 동일 설정: C_miss=2, C_fa=1, P_target=0.5.

    DCF(θ) = c_miss * p_target * FNR(θ) + c_fa * (1-p_target) * FPR(θ)
    minDCF = min_θ DCF(θ) / min(c_miss*p_target, c_fa*(1-p_target))
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    dcf = c_miss * p_target * fnr + c_fa * (1.0 - p_target) * fpr
    min_default = min(c_miss * p_target, c_fa * (1.0 - p_target))
    return float(np.min(dcf) / min_default)


def compute_fpr_fnr(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """FPR/FNR at a given threshold.

    Anti-spoofing 관점 (임동훈 2024 Table 5 기준):
      FPR = real 음성을 fake로 오판하는 비율 (false alarm on bonafide)
      FNR = fake 음성을 real로 오판하는 비율 (miss detection on spoof)

    Note: 본 코드에서 label=1은 real, score=P(real).
      pred >= threshold → real로 분류
      pred <  threshold → fake로 분류
    """
    preds = (scores >= threshold).astype(int)
    real_mask = labels == 1
    fake_mask = labels == 0

    n_real = real_mask.sum()
    n_fake = fake_mask.sum()

    # FPR: real인데 fake로 예측 (pred=0)
    fpr = float((preds[real_mask] == 0).sum() / n_real) if n_real > 0 else 0.0
    # FNR: fake인데 real로 예측 (pred=1)
    fnr = float((preds[fake_mask] == 1).sum() / n_fake) if n_fake > 0 else 0.0

    return {"fpr": fpr, "fnr": fnr}


def compute_eer_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """EER 지점의 threshold를 반환."""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float(thresholds[min(idx, len(thresholds) - 1)])


def compute_fpr_fnr_at_eer(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """EER 임계값에서의 FPR/FNR 분리 반환.

    ASVspoof 학술 관례에 따른 분석. EER threshold에서는
    이론적으로 FPR ≈ FNR ≈ EER이지만, 이산 데이터에서는
    약간 차이날 수 있으며, 그 차이 자체가 데이터 분포 정보를 담는다.
    """
    eer_thr = compute_eer_threshold(labels, scores)
    return compute_fpr_fnr(labels, scores, threshold=eer_thr)


def compute_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    return float((labels == preds).mean())


def compute_all(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """모든 평가 지표를 한 번에 계산하여 반환.

    FPR/FNR은 두 가지 기준으로 보고:
      - fpr, fnr: threshold=0.5 (실용적 배포 기준)
      - fpr_eer, fnr_eer: EER threshold (학술 비교 기준)
    """
    preds = (scores >= 0.5).astype(int)
    fpr_fnr_05 = compute_fpr_fnr(labels, scores, threshold=0.5)
    fpr_fnr_eer = compute_fpr_fnr_at_eer(labels, scores)
    return {
        "eer": compute_eer(labels, scores),
        "mindcf": compute_mindcf(labels, scores),
        "accuracy": compute_accuracy(labels, preds),
        "f1": float(f1_score(labels, preds)),
        "precision": float(precision_score(labels, preds)),
        "recall": float(recall_score(labels, preds)),
        "auc": float(roc_auc_score(labels, scores)),
        "fpr": fpr_fnr_05["fpr"],
        "fnr": fpr_fnr_05["fnr"],
        "fpr_eer": fpr_fnr_eer["fpr"],
        "fnr_eer": fpr_fnr_eer["fnr"],
    }
