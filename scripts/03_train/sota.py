"""SOTA reference numbers from original papers.

각 모델의 원 논문에서 보고된 EER (%) 값.
ASVspoof 2019 LA eval set 기준이며, 우리 테스트셋(T1-T3)과는 다른 데이터셋이므로
직접 비교가 아닌 참고용. T4(ASVspoof 2019 LA eval)를 도입하면 동일 조건 비교 가능.
"""

SOTA = {
    "aasist": {
        "eer": 0.83,
        "dataset": "ASVspoof 2019 LA",
        "paper": "Jung et al., ICASSP 2022",
    },
    "lcnn": {
        "eer": 5.06,
        "dataset": "ASVspoof 2019 LA",
        "paper": "ASVspoof 2019 baseline (LFCC+LCNN)",
    },
    "rawnet2": {
        "eer": 5.13,
        "dataset": "ASVspoof 2019 LA",
        "paper": "Tak et al., ICASSP 2021",
    },
    # XLS-R 기반 (참고)
    "xlsr_aasist": {
        "eer": 0.82,
        "dataset": "ASVspoof 2019 LA",
        "paper": "Tak et al., 2022 (wav2vec2-AASIST)",
    },
}
