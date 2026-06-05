#!/usr/bin/env python3
"""metrics.py 단위 테스트.

합성 데이터로 EER, minDCF, FPR/FNR 계산이 올바른지 검증한다.

사용법:
    python scripts/03_train/test_metrics.py
    python -m pytest scripts/03_train/test_metrics.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import (
    compute_all,
    compute_eer,
    compute_eer_threshold,
    compute_fpr_fnr,
    compute_fpr_fnr_at_eer,
    compute_mindcf,
)


# ---------------------------------------------------------------------------
# 테스트 유틸리티
# ---------------------------------------------------------------------------

def make_perfect_scores(n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """완벽한 분류: real→1.0, fake→0.0."""
    labels = np.array([1] * n + [0] * n)
    scores = np.array([1.0] * n + [0.0] * n)
    return labels, scores


def make_random_scores(n: int = 1000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """랜덤 분류: EER ≈ 50%."""
    rng = np.random.RandomState(seed)
    labels = np.array([1] * n + [0] * n)
    scores = rng.rand(2 * n)
    return labels, scores


def make_good_scores(n: int = 1000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """양호한 분류: real은 높은 점수, fake는 낮은 점수 (약간 겹침)."""
    rng = np.random.RandomState(seed)
    real_scores = rng.beta(8, 2, size=n)   # 평균 ~0.8
    fake_scores = rng.beta(2, 8, size=n)   # 평균 ~0.2
    labels = np.array([1] * n + [0] * n)
    scores = np.concatenate([real_scores, fake_scores])
    return labels, scores


# ---------------------------------------------------------------------------
# EER 테스트
# ---------------------------------------------------------------------------

def test_eer_perfect():
    """완벽 분류 → EER ≈ 0."""
    labels, scores = make_perfect_scores()
    eer = compute_eer(labels, scores)
    assert eer < 0.01, f"완벽 분류인데 EER={eer:.4f} (0에 가까워야 함)"


def test_eer_random():
    """랜덤 분류 → EER ≈ 0.5."""
    labels, scores = make_random_scores()
    eer = compute_eer(labels, scores)
    assert 0.35 < eer < 0.65, f"랜덤 분류인데 EER={eer:.4f} (0.5에 가까워야 함)"


def test_eer_good():
    """양호한 분류 → 0 < EER < 0.2."""
    labels, scores = make_good_scores()
    eer = compute_eer(labels, scores)
    assert 0.0 < eer < 0.2, f"양호 분류인데 EER={eer:.4f}"


def test_eer_range():
    """EER은 항상 [0, 1] 범위."""
    for seed in range(10):
        labels, scores = make_good_scores(seed=seed)
        eer = compute_eer(labels, scores)
        assert 0.0 <= eer <= 1.0, f"EER={eer} 범위 초과"


# ---------------------------------------------------------------------------
# minDCF 테스트
# ---------------------------------------------------------------------------

def test_mindcf_perfect():
    """완벽 분류 → minDCF ≈ 0."""
    labels, scores = make_perfect_scores()
    dcf = compute_mindcf(labels, scores)
    assert dcf < 0.01, f"완벽 분류인데 minDCF={dcf:.4f}"


def test_mindcf_random():
    """랜덤 분류 → minDCF ≈ 1.0 (정규화된 값)."""
    labels, scores = make_random_scores()
    dcf = compute_mindcf(labels, scores)
    assert 0.5 < dcf <= 1.0, f"랜덤 분류인데 minDCF={dcf:.4f} (1.0에 가까워야 함)"


def test_mindcf_range():
    """minDCF는 [0, 1] 범위 (정규화 후)."""
    for seed in range(10):
        labels, scores = make_good_scores(seed=seed)
        dcf = compute_mindcf(labels, scores)
        assert 0.0 <= dcf <= 1.0, f"minDCF={dcf} 범위 초과"


def test_mindcf_cost_params():
    """다른 cost parameter에서도 정상 동작."""
    labels, scores = make_good_scores()
    # ASVspoof 2019 설정
    dcf1 = compute_mindcf(labels, scores, c_miss=1, c_fa=1, p_target=0.01)
    # 임동훈(2024) 설정
    dcf2 = compute_mindcf(labels, scores, c_miss=2, c_fa=1, p_target=0.5)
    assert 0.0 <= dcf1 <= 1.0
    assert 0.0 <= dcf2 <= 1.0


# ---------------------------------------------------------------------------
# FPR/FNR 테스트
# ---------------------------------------------------------------------------

def test_fpr_fnr_perfect():
    """완벽 분류 → FPR=0, FNR=0."""
    labels, scores = make_perfect_scores()
    result = compute_fpr_fnr(labels, scores, threshold=0.5)
    assert result["fpr"] == 0.0, f"FPR={result['fpr']}"
    assert result["fnr"] == 0.0, f"FNR={result['fnr']}"


def test_fpr_fnr_all_fake():
    """모든 점수 0.0 → threshold=0.5 기준 전부 fake 예측 → FPR=1, FNR=0."""
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = compute_fpr_fnr(labels, scores, threshold=0.5)
    assert result["fpr"] == 1.0, f"전부 fake 예측인데 FPR={result['fpr']} (1.0이어야 함)"
    assert result["fnr"] == 0.0, f"fake를 fake로 맞췄는데 FNR={result['fnr']} (0.0이어야 함)"


def test_fpr_fnr_all_real():
    """모든 점수 1.0 → threshold=0.5 기준 전부 real 예측 → FPR=0, FNR=1."""
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    result = compute_fpr_fnr(labels, scores, threshold=0.5)
    assert result["fpr"] == 0.0, f"전부 real 예측인데 FPR={result['fpr']} (0.0이어야 함)"
    assert result["fnr"] == 1.0, f"fake를 real로 오판했는데 FNR={result['fnr']} (1.0이어야 함)"


def test_fpr_fnr_direction():
    """FPR = real→fake 오판, FNR = fake→real 오판 방향 확인."""
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    # real 2개는 높은 점수 (정답), 2개는 낮은 점수 (오판=FPR)
    # fake 3개는 낮은 점수 (정답), 1개는 높은 점수 (오판=FNR)
    scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1, 0.2, 0.3, 0.7])
    result = compute_fpr_fnr(labels, scores, threshold=0.5)
    assert result["fpr"] == 0.5, f"real 4개 중 2개 오판 → FPR=0.5, got {result['fpr']}"
    assert result["fnr"] == 0.25, f"fake 4개 중 1개 오판 → FNR=0.25, got {result['fnr']}"


# ---------------------------------------------------------------------------
# FPR/FNR at EER 테스트
# ---------------------------------------------------------------------------

def test_fpr_fnr_at_eer_symmetry():
    """EER threshold에서 FPR ≈ FNR ≈ EER (이산 데이터라 정확히 같진 않음)."""
    labels, scores = make_good_scores(n=5000)
    eer = compute_eer(labels, scores)
    result = compute_fpr_fnr_at_eer(labels, scores)
    # 이산 데이터에서는 ±2% 오차 허용
    assert abs(result["fpr"] - eer) < 0.02, \
        f"FPR@EER={result['fpr']:.4f} vs EER={eer:.4f} (차이 > 2%)"
    assert abs(result["fnr"] - eer) < 0.02, \
        f"FNR@EER={result['fnr']:.4f} vs EER={eer:.4f} (차이 > 2%)"


def test_eer_threshold_range():
    """EER threshold는 [0, 1] 범위."""
    labels, scores = make_good_scores()
    thr = compute_eer_threshold(labels, scores)
    assert 0.0 <= thr <= 1.0, f"EER threshold={thr} 범위 초과"


# ---------------------------------------------------------------------------
# compute_all 통합 테스트
# ---------------------------------------------------------------------------

def test_compute_all_keys():
    """compute_all이 11개 지표를 모두 반환하는지 확인."""
    labels, scores = make_good_scores()
    result = compute_all(labels, scores)
    expected_keys = {
        "eer", "mindcf", "accuracy", "f1", "precision", "recall", "auc",
        "fpr", "fnr", "fpr_eer", "fnr_eer",
    }
    assert set(result.keys()) == expected_keys, \
        f"누락 키: {expected_keys - set(result.keys())}, 초과 키: {set(result.keys()) - expected_keys}"


def test_compute_all_types():
    """모든 반환값이 float인지 확인."""
    labels, scores = make_good_scores()
    result = compute_all(labels, scores)
    for k, v in result.items():
        assert isinstance(v, float), f"{k}={v} (type={type(v).__name__}, float이어야 함)"


def test_compute_all_ranges():
    """모든 지표가 [0, 1] 범위인지 확인."""
    labels, scores = make_good_scores()
    result = compute_all(labels, scores)
    for k, v in result.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} 범위 초과 [0, 1]"


def test_compute_all_perfect():
    """완벽 분류에서 모든 지표가 최적값인지 확인."""
    labels, scores = make_perfect_scores()
    result = compute_all(labels, scores)
    assert result["eer"] < 0.01
    assert result["mindcf"] < 0.01
    assert result["accuracy"] > 0.99
    assert result["f1"] > 0.99
    assert result["auc"] > 0.99
    assert result["fpr"] < 0.01
    assert result["fnr"] < 0.01


# ---------------------------------------------------------------------------
# 엣지 케이스
# ---------------------------------------------------------------------------

def test_imbalanced_labels():
    """불균형 레이블 (real >> fake) 에서도 크래시 안 남."""
    rng = np.random.RandomState(42)
    labels = np.array([1] * 900 + [0] * 100)
    scores = rng.rand(1000)
    result = compute_all(labels, scores)
    assert len(result) == 11


def test_small_dataset():
    """극소 데이터셋에서도 동작."""
    labels = np.array([1, 0])
    scores = np.array([0.9, 0.1])
    result = compute_all(labels, scores)
    assert result["eer"] < 0.5
    assert result["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    """모든 테스트 실행."""
    import inspect
    test_funcs = [
        (name, obj) for name, obj in globals().items()
        if name.startswith("test_") and inspect.isfunction(obj)
    ]

    passed, failed = 0, 0
    for name, func in sorted(test_funcs):
        try:
            func()
            print(f"  ✓ {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {name}: CRASH — {e}")
            failed += 1

    print(f"\n  {'='*50}")
    print(f"  총 {passed + failed}개 | 통과 {passed} | 실패 {failed}")
    print(f"  {'='*50}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    print("\n  === metrics.py 단위 테스트 ===\n")
    sys.exit(run_all_tests())
