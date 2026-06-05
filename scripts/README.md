# scripts/ — 실행 가이드

한국어 딥페이크 음성 탐지 연구의 전체 파이프라인. Phase 1(다운로드/전처리) → Phase 2(모델 세팅) → Phase 3(핵심 실험) → Phase 4(지표 보강) → Phase 5(강건성 실험) → Phase 6(시각화) 순서로 실행한다.

## 디렉토리 구조

```
scripts/
├── 01_download/                    ← 데이터셋 다운로드
│   ├── download_mlaad.py               MLAAD v9 (다국어 fake, 51개 언어)
│   ├── download_mailabs.py             M-AILABS (다국어 real, 9개 언어)
│   ├── download_speechfake.py          SpeechFake-MD (다국어 fake+real)
│   ├── download_koaad.py               KoAAD (한국어 fake+real)
│   └── download_musan.py               MUSAN (노이즈 증강용)
│
├── 02_preprocess/                  ← 전처리 및 데이터 분할
│   ├── preprocess.py                   스캔 → 샘플링 → 리샘플링 → 품질검증
│   └── build_splits.py                 C1/C2/C3 조합 + T1/T2/T3 평가셋 생성
│
├── 03_train/                       ← 학습 및 평가
│   ├── train.py                        학습 + 평가 메인 스크립트
│   ├── build_results.py                체크포인트 재평가 → results.csv
│   ├── eval_noise.py                   테스트 시 노이즈 강건성 평가
│   ├── dataset.py                      AudioDataset (증강 지원)
│   ├── metrics.py                      EER, minDCF, FPR/FNR, FPR/FNR@EER, Accuracy, F1, AUC
│   ├── test_metrics.py                 metrics.py 단위 테스트 (20개)
│   ├── augment.py                      노이즈/코덱 증강 유틸리티
│   ├── sota.py                         ASVspoof 2019 SOTA 레퍼런스
│   ├── merge_manifests.py              C_all manifest 병합
│   ├── verify_models.py                모델 forward/backward 검증
│   ├── launch_tb.py                    TensorBoard 런처
│   ├── configs/                        실험별 YAML (23개)
│   │   ├── {model}_{combo}.yaml            Phase 3 핵심 (13개)
│   │   ├── {model}_call_aug_{type}.yaml    Phase 5-B 증강 (6개)
│   │   └── {model}_call_{length}.yaml      Phase 5-C 길이 (4개)
│   └── models/                         모델 구현
│       ├── aasist.py                       AASIST (229K params)
│       ├── specrnet.py                     SpecRNet (416K params)
│       ├── lcnn.py                         LCNN (1.37M params)
│       ├── rawnet2.py                      RawNet2 (18.5M params)
│       └── sincconv.py                     SincConv (공유 모듈)
│
├── 04_analyze/                     ← 결과 분석 및 시각화
│   ├── analyze.py                      결과 분석 + SOTA 비교 출력
│   ├── compare_lim.py                  임동훈(2024) 직접 비교표
│   └── visualize.py                    논문용 그래프(G1~G4) + 테이블(T1~T4)
│
└── utils/
    └── list_data.py                    디렉토리 구조 탐색
```

## 필요 패키지

```bash
# 다운로드
pip install requests tqdm huggingface_hub gdown modelscope

# 전처리
pip install soundfile librosa numpy tqdm pandas scikit-learn

# 학습/평가
pip install torch torchaudio tensorboard pyyaml scipy

# 시각화
pip install matplotlib
```

---

## 실행 순서

### Phase 1: 데이터 다운로드

```bash
# 1-1. KoAAD (한국어 fake+real, ~10GB)
python scripts/01_download/download_koaad.py -o data/koaad

# 1-2. MLAAD v9 (다국어 fake, ~109GB)
python scripts/01_download/download_mlaad.py -o data/mlaad
# 한국어만: --languages ko

# 1-3. M-AILABS (다국어 real, ~127GB)
python scripts/01_download/download_mailabs.py -o data/mailabs
# 필요 언어만: --languages en_US de_DE fr_FR

# 1-4. SpeechFake-MD (다국어 fake+real)
python scripts/01_download/download_speechfake.py -o data/speechfake --method modelscope

# 1-5. AI-Hub 한국어 real (수동)
#   → https://www.aihub.or.kr 에서 "감성 및 발화스타일 동시고려 음성합성 데이터" 다운로드
#   → data/aihub/ 에 배치
```

### Phase 2: 전처리 및 데이터 분할

```bash
# 2-1. 메타데이터 스캔
python scripts/02_preprocess/preprocess.py scan \
    --data-root ./data --meta-dir metadata

# 2-2. 스마트 샘플링 (real:fake 비율 조정)
python scripts/02_preprocess/preprocess.py sample \
    --meta-dir metadata --seed 42 --buffer 1.3

# 2-3. 리샘플링 + 품질검증 (16kHz mono WAV)
python scripts/02_preprocess/preprocess.py process \
    --workers 8 --output-root processed --meta-dir metadata

# 2-4. 통계 리포트
python scripts/02_preprocess/preprocess.py report --meta-dir metadata

# 2-5. 데이터 분할 (C1/C2/C3 + T1/T2/T3)
python scripts/02_preprocess/build_splits.py \
    --meta-dir metadata --output-dir manifests --seed 42

# 2-6. C_all manifest 병합
python scripts/03_train/merge_manifests.py
```

**전처리 파이프라인:**
```
원시 데이터 (1,512,236개)
  → 스마트 샘플링 (→ 1,013,041개)
  → 16kHz mono WAV 리샘플링
  → 품질 필터링: 무음(>95%), 클리핑(>1%), 짧은 파일(<0.5s), 손상 제거
  → 최종 (→ 1,002,528개)
```

**데이터 조합:**

| 조합 | 구성 | 한국어 | 학습 수 |
|------|------|:------:|--------:|
| C1 | MLAAD v9 (ko 제외) + M-AILABS | X | 373,090 |
| C2 | MLAAD v9 (전체) + M-AILABS + SpeechFake ko | O | 555,802 |
| C3 | SpeechFake ko + KoAAD | O only | 139,068 |
| C_all | C1 + C2 + C3 통합 | O | 1,067,960 |

**평가셋 (전 실험 고정):**

| 셋 | 내용 | 샘플 수 |
|----|------|--------:|
| T1 | SpeechFake ko test | 982 |
| T2 | KoAAD holdout | 26,977 |
| T3 | MLAAD test (ko 제외) | 116,544 |

### Phase 3: 핵심 실험 (13개) ✅ 완료

```bash
# 모델 검증
python scripts/03_train/verify_models.py

# 단일 실험
python scripts/03_train/train.py --config scripts/03_train/configs/aasist_c1.yaml

# 전체 실행 (13개 순차)
python scripts/03_train/train.py --all

# 중단 후 재개
python scripts/03_train/train.py --all --resume
```

**Phase 3-1**: 3개 모델 × 3개 조합 = 9개 실험
```
aasist_c1, aasist_c2, aasist_c3
lcnn_c1,   lcnn_c2,   lcnn_c3
rawnet2_c1, rawnet2_c2, rawnet2_c3
```

**Phase 3-2**: 4개 모델 × C_all = 4개 실험
```
aasist_call, specrnet_call, lcnn_call, rawnet2_call
```

### Phase 4: 평가 지표 보강 ← 현재 단계

기존 13개 체크포인트를 재평가하여 minDCF, FPR/FNR을 추가 산출한다.

```bash
# 4-1. 전체 재평가 (minDCF + FPR/FNR 포함)
python scripts/03_train/build_results.py --all --output experiments/results_v2.csv

# 4-2. 임동훈(2024) 비교표 생성
python scripts/04_analyze/compare_lim.py --results experiments/results_v2.csv

# 4-3. 결과 분석 출력
python scripts/04_analyze/analyze.py --results experiments/results_v2.csv
```

**평가 지표:**

| 지표 | 설명 | 비교 대상 |
|------|------|----------|
| EER | FAR=FRR 교차점 | 임동훈, SpeechFake, ASVspoof |
| minDCF | 비용 가중 오류율 (C_miss=2, C_fa=1) | 임동훈, ASVspoof |
| FPR | real→fake 오판률 (threshold=0.5) | 임동훈 Table 5 |
| FNR | fake→real 오판률 (threshold=0.5) | 임동훈 Table 5 |
| FPR@EER | real→fake 오판률 (EER threshold) | ASVspoof 학술 관례 |
| FNR@EER | fake→real 오판률 (EER threshold) | ASVspoof 학술 관례 |
| Accuracy / F1 / AUC | 표준 분류 지표 | 임동훈 |
| 추론 시간 / 파라미터 / FLOPs | 경량화 지표 | — |

### Phase 5: 강건성 실험 (~20개)

#### 5-A. MUSAN 다운로드

```bash
# noise 카테고리만 (~1GB)
python scripts/01_download/download_musan.py -o data/musan --categories noise
```

#### 5-B. 테스트 시 노이즈 평가 (실험 A)

기존 체크포인트로 평가만 수행. 재학습 없음.

```bash
# 전체 (5조건 × 2모델 × 3테스트셋 = 30개 평가)
python scripts/03_train/eval_noise.py --musan-dir data/musan/musan

# 특정 조건/모델만
python scripts/03_train/eval_noise.py --musan-dir data/musan/musan \
    --conditions clean gaussian_001 musan_10db \
    --models specrnet lcnn \
    --eval-sets T1 T2
```

**노이즈 조건:**

| 조건 | 설명 | 비교 대상 |
|------|------|----------|
| clean | 원본 (대조군) | — |
| gaussian_001 | Gaussian α=0.001 | 임동훈 Table 3 |
| musan_20db | MUSAN SNR 20dB | 가벼운 잡음 |
| musan_10db | MUSAN SNR 10dB | 보통 잡음 |
| musan_5db | MUSAN SNR 5dB | 악조건 |

#### 5-C. 학습 시 증강 (실험 B)

C_all 데이터에 증강을 적용하여 재학습. SpecRNet + LCNN 각각 수행.

```bash
# B-2: MUSAN 증강 (1순위)
python scripts/03_train/train.py --config scripts/03_train/configs/specrnet_call_aug_musan.yaml
python scripts/03_train/train.py --config scripts/03_train/configs/lcnn_call_aug_musan.yaml

# B-3: Codec 증강 (2순위)
python scripts/03_train/train.py --config scripts/03_train/configs/specrnet_call_aug_codec.yaml
python scripts/03_train/train.py --config scripts/03_train/configs/lcnn_call_aug_codec.yaml

# B-4: MUSAN + Codec (2순위)
python scripts/03_train/train.py --config scripts/03_train/configs/specrnet_call_aug_both.yaml
python scripts/03_train/train.py --config scripts/03_train/configs/lcnn_call_aug_both.yaml
```

#### 5-D. 발화 길이별 성능 (실험 C, 2순위)

```bash
# C-1: 2초
python scripts/03_train/train.py --config scripts/03_train/configs/specrnet_call_2sec.yaml
python scripts/03_train/train.py --config scripts/03_train/configs/lcnn_call_2sec.yaml

# C-3: 8초
python scripts/03_train/train.py --config scripts/03_train/configs/specrnet_call_8sec.yaml
python scripts/03_train/train.py --config scripts/03_train/configs/lcnn_call_8sec.yaml
```

### Phase 6: 분석 및 시각화

```bash
# 전체 생성 (그래프 4개 + 테이블 4개)
python scripts/04_analyze/visualize.py --all --results experiments/results_v2.csv

# 개별 그래프
python scripts/04_analyze/visualize.py --graph G1   # 데이터 커버리지별 EER
python scripts/04_analyze/visualize.py --graph G2   # 성능-파라미터 트레이드오프
python scripts/04_analyze/visualize.py --graph G3   # SNR별 성능 하락
python scripts/04_analyze/visualize.py --graph G4   # Phase 1→2 개선

# 테이블만
python scripts/04_analyze/visualize.py --tables
```

**출력물:**

| 그래프 | 내용 |
|--------|------|
| G1 | 데이터 커버리지별 EER (Grouped bar, 모델×조합) |
| G2 | 성능-파라미터 트레이드오프 (Scatter, C_all 기준) |
| G3 | SNR별 성능 하락 (Line chart) |
| G4 | Phase 1→2 개선 (Before/After bar) |

| 테이블 | 내용 |
|--------|------|
| T1 | 전체 13개 실험 결과 (EER, minDCF) |
| T2 | C_all 4개 모델 비교 (성능 + 경량화) |
| T3 | 임동훈(2024) 직접 비교 |
| T4 | 노이즈 강건성표 |

---

## 출력 디렉토리 구조

```
data/                               ← 원본 데이터
├── mlaad/fake/{lang}/{tts}/*.wav
├── mailabs/{locale}/{gender}/{speaker}/*.wav
├── speechfake/{MD,Real}/...
├── koaad/{tts_type}/*.wav
├── aihub/...
└── musan/musan/{noise,speech,music}/*.wav

metadata/                           ← 전처리 메타데이터
├── raw_metadata.csv
├── sampled_metadata.csv
└── processed_metadata.csv

processed/                          ← 전처리된 오디오 (16kHz mono)
├── koaad/...
├── mailabs/...
├── mlaad/...
├── speechfake-md/...
└── speechfake-real/...

manifests/                          ← 학습/평가 데이터 분할
├── C1/{train,dev}.csv
├── C2/{train,dev}.csv
├── C3/{train,dev}.csv
├── C_all/{train,dev}.csv
├── T1_speechfake_ko_test.csv
├── T2_koaad_holdout.csv
└── T3_mlaad_test.csv

experiments/                        ← 실험 결과
├── checkpoints/{experiment}_best.pt
├── logs/{experiment}.jsonl
├── tb_logs/{experiment}/
├── results.csv                         Phase 3 결과
├── results_v2.csv                      Phase 4 결과 (minDCF/FPR/FNR 포함)
├── noise_robustness.csv                Phase 5-A 노이즈 강건성
├── lim_comparison.csv                  임동훈(2024) 비교표
├── figures/G{1,2,3,4}_{name}.pdf       Phase 6 그래프
└── tables/T{1,2,3,4}_{name}.md         Phase 6 테이블
```

## YAML 설정

**기본 구조:**

```yaml
experiment: specrnet_call            # 실험 이름 (체크포인트 파일명)
data_combo: C_all                    # 데이터 조합

model:
  name: specrnet                     # aasist | specrnet | lcnn | rawnet2

data:
  train_manifest: manifests/C_all/train.csv
  dev_manifest: manifests/C_all/dev.csv
  eval_manifests:
    T1: manifests/T1_speechfake_ko_test.csv
    T2: manifests/T2_koaad_holdout.csv
    T3: manifests/T3_mlaad_test.csv
  num_samples: 64000                 # 4초 × 16kHz (2초=32000, 8초=128000)
  num_workers: 4

training:
  optimizer: adam
  lr: 0.0001
  batch_size: 64                     # AASIST는 16, 8초는 32
  epochs: 50
  early_stopping_patience: 10
  weight_decay: 0.0001
  class_weights: [1.0, 1.0]

# 증강 (선택, 없으면 증강 없음)
augmentation:
  musan_dir: data/musan/musan        # MUSAN 경로
  musan_snr_range: [10, 20]          # SNR 범위 (dB)
  musan_prob: 0.5                    # 적용 확률
  codec: mp3                         # 코덱 종류
  codec_bitrate: "64k"               # 비트레이트
  codec_prob: 0.3                    # 적용 확률

seed: 42
output_root: experiments
```

**설정 파일 목록 (23개):**

| Phase | 설정 파일 | 모델 | 데이터 | 비고 |
|-------|----------|------|--------|------|
| 3-1 | aasist_c1~c3, lcnn_c1~c3, rawnet2_c1~c3 | 3종 | C1/C2/C3 | 9개 |
| 3-2 | aasist_call, specrnet_call, lcnn_call, rawnet2_call | 4종 | C_all | 4개 |
| 5-B | specrnet_call_aug_musan, lcnn_call_aug_musan | 2종 | C_all+MUSAN | 2개 |
| 5-B | specrnet_call_aug_codec, lcnn_call_aug_codec | 2종 | C_all+Codec | 2개 |
| 5-B | specrnet_call_aug_both, lcnn_call_aug_both | 2종 | C_all+MUSAN+Codec | 2개 |
| 5-C | specrnet_call_2sec, lcnn_call_2sec | 2종 | C_all, 2초 | 2개 |
| 5-C | specrnet_call_8sec, lcnn_call_8sec | 2종 | C_all, 8초 | 2개 |

## GPU 최적화

- BF16 mixed precision (`torch.amp.autocast`)
- TF32 matmul/cuDNN (`torch.backends.cuda.matmul.allow_tf32`)
- `torch.compile` (PyTorch 2.0+)
- `pin_memory` + `non_blocking` + `persistent_workers`

## 라이선스

| 데이터셋 | 라이선스 |
|----------|----------|
| MLAAD v9 | CC-BY-NC 4.0 |
| M-AILABS | BSD 3-Clause |
| SpeechFake | CC BY 4.0 |
| KoAAD | 연구 목적 |
| AI-Hub | 연구 목적 |
| MUSAN | CC BY 4.0 |
