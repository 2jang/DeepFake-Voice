# 03_train — 학습 및 평가

Deepfake 음성 탐지 모델 학습, 평가, SOTA 비교를 수행하는 스크립트 모음.

## 디렉토리 구조

```
03_train/
├── train.py              # 학습/평가 메인 스크립트 (--all 지원)
├── analyze.py            # 결과 분석 + SOTA 비교
├── sota.py               # SOTA 레퍼런스 수치 (원 논문 기준)
├── dataset.py            # AudioDataset (WAV → 고정 길이 텐서)
├── metrics.py            # EER, Accuracy, F1, Precision, Recall, AUC
├── verify_models.py      # 모델 forward/backward 검증
├── configs/              # 실험별 YAML 설정 (3모델 × 3조합 = 9개)
└── models/
    ├── aasist.py          # AASIST (~297K params)
    ├── lcnn.py            # LCNN (~1.4M params)
    ├── rawnet2.py         # RawNet2 (~17M params)
    └── sincconv.py        # SincConv (AASIST, RawNet2 공유)
```

## 모델

| 모델 | 파라미터 | 입력 | 특징 |
|------|---------|------|------|
| AASIST | ~297K | raw waveform | SincConv + 2D ResBlock + Graph Attention |
| LCNN | ~1.4M | raw waveform → log power spectrogram | MFM activation + CNN |
| RawNet2 | ~17M | raw waveform | SincConv + 1D ResBlock + GRU |

## 사용법

### 학습

```bash
# 단일 실험
python scripts/03_train/train.py --config scripts/03_train/configs/aasist_c1.yaml

# 전체 9개 실험 순차 실행
python scripts/03_train/train.py --all

# epoch 오버라이드
python scripts/03_train/train.py --all --epochs 5

# CPU 강제
python scripts/03_train/train.py --config scripts/03_train/configs/aasist_c1.yaml --device cpu
```

### 결과 분석

```bash
python scripts/03_train/analyze.py
python scripts/03_train/analyze.py --results path/to/results.csv
```

### 모델 검증

```bash
python scripts/03_train/verify_models.py
```

## 설정 (YAML)

```yaml
experiment: aasist_c1
data_combo: C1

model:
  name: aasist          # aasist | lcnn | rawnet2

data:
  train_manifest: manifests/C1/train.csv
  dev_manifest: manifests/C1/dev.csv
  eval_manifests:
    T1: manifests/T1_speechfake_ko_test.csv
    T2: manifests/T2_koaad_holdout.csv
    T3: manifests/T3_mlaad_test.csv
  num_samples: 64000    # 4초 (16kHz)
  num_workers: 4

training:
  lr: 0.0001
  batch_size: 16
  epochs: 50
  early_stopping_patience: 10
  weight_decay: 0.0001
  class_weights: [1.0, 1.0]
```

## 데이터 조합

| 조합 | 학습 데이터 |
|------|------------|
| C1 | KoAAD + SpeechFake(ko) |
| C2 | C1 + M-AILABS(en/de) |
| C3 | C2 + MLAAD |

## 평가 지표

### 탐지 성능

| 지표 | 설명 |
|------|------|
| EER | Equal Error Rate (FAR == FRR). 낮을수록 좋음 |
| Accuracy | 임계값 0.5 기준 정확도 |
| F1 Score | Precision과 Recall의 조화평균 |
| Precision | 양성 예측 중 실제 양성 비율 |
| Recall | 실제 양성 중 탐지 비율 |
| AUC | ROC 곡선 아래 면적 |

### 경량화 성능

| 지표 | 설명 |
|------|------|
| Inference Time | 단일 샘플 추론 시간 (ms) |
| Throughput | 초당 처리 샘플 수 (samples/sec) |
| FLOPs | 연산량 (M) |
| Model Size | 모델 크기 (MB) |
| Peak Memory | 추론 시 최대 GPU 메모리 사용량 (MB) |
| Parameters | 학습 가능한 파라미터 수 |

## SOTA 레퍼런스

ASVspoof 2019 LA eval set 기준 원 논문 수치:

| 모델 | EER (%) | 출처 |
|------|---------|------|
| AASIST | 0.83 | Jung et al., ICASSP 2022 |
| LCNN | 5.06 | ASVspoof 2019 baseline (LFCC+LCNN) |
| RawNet2 | 5.13 | Tak et al., ICASSP 2021 |

본 프로젝트의 평가셋(T1/T2/T3)과는 데이터셋이 다르므로 직접 비교가 아닌 참고용.

## 출력

```
experiments/
├── checkpoints/          # {experiment}_best.pt
├── logs/                 # {experiment}.jsonl (epoch별 전체 지표)
└── results.csv           # 전체 실험 결과 + 경량화 지표
```

## GPU 최적화

- BF16 mixed precision
- TF32 matmul/cuDNN
- torch.compile
- pin_memory + non_blocking
- persistent_workers + prefetch_factor=4
