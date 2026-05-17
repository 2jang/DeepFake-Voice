# scripts/

한국어 딥페이크 음성 탐지 연구 파이프라인 스크립트 모음.

## 디렉토리 구조

```
scripts/
├── 01_download/        ← Phase 1.1: 데이터셋 다운로드
│   ├── download_mlaad.py
│   ├── download_mailabs.py
│   ├── download_speechfake.py
│   ├── download_koaad.py
│   └── download_musan.py
│
├── 02_preprocess/      ← Phase 1.2–1.4: 전처리 및 데이터 분할
│   ├── preprocess.py        (스캔 → 리샘플링 → 품질검증)
│   └── build_splits.py      (C1/C2/C3 조합 + T1/T2/T3 평가셋)
│
└── utils/              ← 범용 유틸리티
    └── list_data.py         (디렉토리 구조 탐색)
```

## 실행 순서

```
Phase 1.1  다운로드      ──→  Phase 1.2  전처리       ──→  Phase 1.3-1.4  분할
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

01_download/             02_preprocess/             02_preprocess/
download_*.py            preprocess.py scan         build_splits.py
                         preprocess.py process
                         preprocess.py report
```

## 빠른 시작

```bash
# 1. 데이터 다운로드 (Phase 1.1)
python scripts/01_download/download_mlaad.py -o data/mlaad --languages ko
python scripts/01_download/download_musan.py -o data/musan --categories noise

# 2. 메타데이터 스캔 (Phase 1.2)
python scripts/02_preprocess/preprocess.py scan --data-root ./data --speechfake-root D:/speechfake

# 3. 리샘플링 + 품질검증 (Phase 1.2)
python scripts/02_preprocess/preprocess.py process --workers 8

# 4. 데이터 분할 (Phase 1.3-1.4)
python scripts/02_preprocess/build_splits.py --meta-dir metadata --output-dir manifests
```

## 필요 패키지

```bash
# 다운로드
pip install requests tqdm huggingface_hub gdown modelscope

# 전처리
pip install soundfile librosa numpy tqdm pandas scikit-learn
```

## 각 폴더 상세

- [`01_download/README.md`](01_download/README.md) — 데이터셋별 다운로드 방법
- [`02_preprocess/README.md`](02_preprocess/README.md) — 전처리 파이프라인 상세
- [`utils/README.md`](utils/README.md) — 유틸리티 도구
