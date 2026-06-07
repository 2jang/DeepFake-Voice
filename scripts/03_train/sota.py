"""SOTA reference numbers — 원 논문 + CD-ADD 벤치마크 리더보드.

리더보드 출처: CD-ADD benchmark (Li et al., EMNLP 2024)
모든 수치는 EER (%) — 낮을수록 좋음.

벤치마크 매핑:
    T4 = ASVspoof 2019 LA eval
    T5 = ASVspoof 2021 DF eval
    T6 = ASVspoof 2021 LA eval
    T7 = CD-ADD
    T8 = InTheWild
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Per-model SOTA: 원 논문 / 리더보드에서 보고된 EER (%)
# ---------------------------------------------------------------------------

SOTA: dict[str, dict] = {
    "aasist": {
        "paper": "Jung et al., ICASSP 2022",
        "leaderboard_rank": 6,
        "benchmarks": {
            "T4": 0.83,
            "T5": 17.04,
            "T6": 12.35,
            "T7": 51.05,
            "T8": 43.02,
        },
        "mean": 24.86,
    },
    "lcnn": {
        "paper": "ASVspoof 2019 baseline (LFCC+LCNN)",
        "benchmarks": {
            "T4": 5.06,
        },
    },
    "rawnet2": {
        "paper": "Tak et al., ICASSP 2021",
        "benchmarks": {
            "T4": 5.13,
        },
    },
    # XLS-R 기반 — W2V2-AASIST (리더보드 #3)
    "xlsr_aasist": {
        "paper": "Tak et al., 2022 (wav2vec2-AASIST)",
        "leaderboard_rank": 3,
        "benchmarks": {
            "T4": 0.22,
            "T5": 8.32,
            "T6": 8.11,
            "T7": 38.57,
            "T8": 11.22,
        },
        "mean": 13.29,
    },
}

# Backward compat: 기존 코드에서 SOTA[model]["eer"] / ["dataset"] 접근 지원
for _name, _ref in SOTA.items():
    if "T4" in _ref.get("benchmarks", {}):
        _ref.setdefault("eer", _ref["benchmarks"]["T4"])
        _ref.setdefault("dataset", "ASVspoof 2019 LA")


# ---------------------------------------------------------------------------
# CD-ADD 벤치마크 리더보드 (Top 10)
# ---------------------------------------------------------------------------

LEADERBOARD: list[dict] = [
    {"rank": 1,  "system": "XLSR-SLS",            "mean": 5.76,  "T4": 0.23, "T5": 3.93,  "T6": 7.39,  "T7": 9.81,  "T8": 7.46},
    {"rank": 2,  "system": "Nes2Net",              "mean": 7.78,  "T4": 0.13, "T5": 3.61,  "T6": 6.14,  "T7": 20.55, "T8": 8.48},
    {"rank": 3,  "system": "W2V2-AASIST",          "mean": 13.29, "T4": 0.22, "T5": 8.32,  "T6": 8.11,  "T7": 38.57, "T8": 11.22},
    {"rank": 4,  "system": "Whisper-MFCC-MesoNet",  "mean": 13.58, "T4": 5.83, "T5": 0.46,  "T6": 15.96, "T7": 18.90, "T8": 26.72},
    {"rank": 5,  "system": "RawTFNet",             "mean": 23.31, "T4": 1.99, "T5": 15.16, "T6": 8.03,  "T7": 52.85, "T8": 38.51},
    {"rank": 6,  "system": "AASIST",               "mean": 24.86, "T4": 0.83, "T5": 17.04, "T6": 12.35, "T7": 51.05, "T8": 43.02},
    {"rank": 7,  "system": "AASIST-L",             "mean": 25.05, "T4": 0.99, "T5": 15.96, "T6": 13.15, "T7": 50.72, "T8": 44.45},
    {"rank": 8,  "system": "AASIST3",              "mean": 26.14, "T4": 9.44, "T5": 28.73, "T6": 32.06, "T7": 30.73, "T8": 29.73},
    {"rank": 9,  "system": "Res2TCNGuard",         "mean": 28.16, "T4": 1.50, "T5": 17.02, "T6": 13.67, "T7": 56.10, "T8": 52.52},
    {"rank": 10, "system": "ResCapsGuard",          "mean": 29.61, "T4": 1.86, "T5": 17.00, "T6": 18.70, "T7": 54.55, "T8": 55.92},
]


# ---------------------------------------------------------------------------
# Per-benchmark best known EER (%)
# ---------------------------------------------------------------------------

BENCHMARK_BEST: dict[str, dict] = {
    "T4": {"eer": 0.13, "system": "Nes2Net"},
    "T5": {"eer": 0.46, "system": "Whisper-MFCC-MesoNet"},
    "T6": {"eer": 6.14, "system": "Nes2Net"},
    "T7": {"eer": 9.81, "system": "XLSR-SLS"},
    "T8": {"eer": 7.46, "system": "XLSR-SLS"},
}


BENCHMARK_NAMES: dict[str, str] = {
    "T4": "ASVspoof 2019 LA",
    "T5": "ASVspoof 2021 DF",
    "T6": "ASVspoof 2021 LA",
    "T7": "CD-ADD",
    "T8": "InTheWild",
}
