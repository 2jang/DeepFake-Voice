# 03_train — 학습 및 평가

Deepfake 음성 탐지 모델 학습, 평가, 추론 속도 측정을 수행하는 스크립트 모음.

## 디렉토리 구조

```
03_train/
├── train.py              # 학습/평가 메인 스크립트
├── dataset.py            # AudioDataset (WAV → 고정 길이 텐서)
├── metrics.py            # EER, Accuracy 계산
├── verify_models.py      # 모델 forward/backward 검증
├── configs/              # 실험별 YAML 설정
│   ├── aasist_c1.yaml
│   ├── lcnn_c1.yaml
│   └── rawnet2_c1.yaml   ... (총 9개: 3모델 × 3데이터 조합)
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

### 로컬 학습

```bash
# 단일 실험
python scripts/03_train/train.py --config scripts/03_train/configs/aasist_c1.yaml

# epoch 수 오버라이드
python scripts/03_train/train.py --config scripts/03_train/configs/aasist_c1.yaml --epochs 5

# CPU 강제
python scripts/03_train/train.py --config scripts/03_train/configs/aasist_c1.yaml --device cpu
```

### 모델 검증

```bash
python scripts/03_train/verify_models.py
```

3개 모델의 forward/backward pass가 정상 동작하는지 확인.

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

- **EER (Equal Error Rate)**: FAR == FRR인 지점. 낮을수록 좋음
- **Accuracy**: 임계값 0.5 기준 정확도

## 출력

```
experiments/
├── checkpoints/          # {experiment}_best.pt
├── logs/                 # {experiment}.jsonl (epoch별 기록)
└── results.csv           # 전체 실험 결과 요약
```

## GPU 최적화 (train.py 기본 적용)

- BF16 mixed precision (GradScaler 불필요)
- TF32 matmul/cuDNN 가속
- torch.compile 그래프 컴파일
- pin_memory + non_blocking 전송
- persistent_workers + prefetch_factor=4
