# scripts/ — 실행 가이드

한국어 딥페이크 음성 탐지 연구의 전체 파이프라인. Phase 1(다운로드/전처리) → Phase 2(모델 세팅) → Phase 3(핵심 실험) → Phase 4(강건성 실험) → Phase 5(지표 보강) → Phase 6(시각화) 순서로 실행한다.

## 디렉토리 구조

```
scripts/
├── 01_download/                    ← 데이터셋 다운로드
│   ├── download_mlaad.py               MLAAD v9 (다국어 fake, 51개 언어)
│   ├── download_mailabs.py             M-AILABS (다국어 real, 9개 언어)
│   ├── download_speechfake.py          SpeechFake-MD (다국어 fake+real)
│   ├── download_koaad.py               KoAAD (한국어 fake+real)
│   ├── download_musan.py               MUSAN (노이즈 증강용)
│   ├── download_asvspoof19.py          ASVspoof 2019 LA (T4)
│   ├── download_asvspoof21.py          ASVspoof 2021 DF+LA (T5, T6)
│   ├── download_cdadd.py               CD-ADD (T7)
│   └── download_inthewild.py           InTheWild (T8)
│
├── 02_preprocess/                  ← 전처리 및 데이터 분할
│   ├── preprocess.py                   스캔 → 샘플링 → 리샘플링 → 품질검증
│   ├── build_splits.py                 C1/C2/C3 조합 + T1/T2/T3 평가셋 생성
│   └── prepare_benchmarks.py           T4~T8 외부 벤치마크 manifest 빌더
│
├── 03_train/                       ← 학습 및 평가
│   ├── train.py                        학습 + 평가 메인 스크립트
│   ├── build_results.py                체크포인트 재평가 → results_v2.csv
│   ├── eval_noise.py                   테스트 시 노이즈 강건성 평가 (2×2 매트릭스)
│   ├── dataset.py                      AudioDataset (증강 지원)
│   ├── metrics.py                      EER, minDCF, FPR/FNR, FPR/FNR@EER, Accuracy, F1, AUC
│   ├── test_metrics.py                 metrics.py 단위 테스트 (20개)
│   ├── augment.py                      노이즈/코덱 증강 유틸리티
│   ├── sota.py                         ASVspoof 2019 SOTA 레퍼런스
│   ├── merge_manifests.py              C_all manifest 병합
│   ├── verify_models.py                모델 forward/backward 검증
│   ├── launch_tb.py                    TensorBoard 런처
│   ├── configs/                        실험별 YAML (25개)
│   │   ├── {model}_{combo}.yaml            Phase 3 핵심 (13개)
│   │   ├── {model}_call_aug_{type}.yaml    Phase 4-B 증강 (6개)
│   │   ├── {model}_call_{length}.yaml      Phase 4-C 길이 (4개)
│   │   └── xlsr_{head}_call.yaml           Phase 4-D SSL (2개)
│   └── models/                         모델 구현
│       ├── aasist.py                       AASIST (229K params)
│       ├── specrnet.py                     SpecRNet (416K params)
│       ├── lcnn.py                         LCNN (1.37M params)
│       ├── rawnet2.py                      RawNet2 (18.5M params)
│       ├── xlsr_head.py                    XLS-R SSL (296K/2K trainable, 300M frozen)
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

루트 `requirements.txt` 하나로 전체 파이프라인(다운로드·전처리·학습·시각화) 의존성을 설치한다.

```bash
pip install -r requirements.txt
```

> - GPU(CUDA) 환경에서는 `torch`를 [PyTorch 공식 안내](https://pytorch.org/get-started/locally/)에 따라 CUDA 빌드로 먼저 설치한 뒤 나머지를 설치하는 것을 권장한다.
> - 코덱 증강은 PyAV(`av`)가 ffmpeg를 번들하므로 별도 시스템 ffmpeg 설치가 필요 없다.
> - SSL 베이스라인(Phase 4-D)은 `transformers` 패키지가 필요하며, XLS-R 모델(~1.2GB)은 첫 실행 시 자동 다운로드된다.

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

# 1-6. MUSAN (노이즈 증강용, ~1GB)
python scripts/01_download/download_musan.py -o data/musan --categories noise

# 1-7. 외부 벤치마크 (T4~T8)
python scripts/01_download/download_asvspoof19.py -o data/asvspoof19 --eval-only
python scripts/01_download/download_asvspoof21.py -o data/asvspoof21
python scripts/01_download/download_cdadd.py -o data/cdadd
python scripts/01_download/download_inthewild.py -o data/inthewild
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

# 2-7. 외부 벤치마크 manifest 빌더 (T4~T8)
python scripts/02_preprocess/prepare_benchmarks.py --all
# 개별: python scripts/02_preprocess/prepare_benchmarks.py asvspoof19 inthewild
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

**평가셋:**

| 셋 | 내용 | 샘플 수 | 유형 |
|----|------|--------:|------|
| T1 | SpeechFake ko test | 982 | 내부 |
| T2 | KoAAD holdout | 26,977 | 내부 |
| T3 | MLAAD test (ko 제외) | 116,544 | 내부 |
| T4 | ASVspoof 2019 LA eval | 71,237 | 외부 벤치마크 |
| T5 | ASVspoof 2021 DF eval | 593,253 | 외부 벤치마크 |
| T6 | ASVspoof 2021 LA eval | — | 외부 벤치마크 |
| T7 | CD-ADD | 300h+ | 외부 벤치마크 |
| T8 | InTheWild | 37.9h | 외부 벤치마크 |

### Phase 3: 핵심 실험 (13개) ✅ 완료

```bash
# 모델 검증
python scripts/03_train/verify_models.py

# 단일 실험
python scripts/03_train/train.py --config scripts/03_train/configs/aasist_c1.yaml

# 전체 실행 (configs/ 전체 순차 — 이미 완료된 실험은 자동 skip)
python scripts/03_train/train.py --all

# 중단된 실험만 이어서 재개
python scripts/03_train/train.py --all --resume

# 완료된 실험도 강제 재학습
python scripts/03_train/train.py --all --force
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

### Phase 4: 강건성 실험 (~20개) ← 현재 단계

```bash
# Phase 4 학습 실험 일괄 (증강 6 + 길이 4 + SSL 2 = 12개)
python scripts/03_train/train.py --phase4

# 서브셋 선택
python scripts/03_train/train.py --phase4 aug          # 증강 6개만
python scripts/03_train/train.py --phase4 len          # 길이 4개만
python scripts/03_train/train.py --phase4 ssl          # SSL 2개만
python scripts/03_train/train.py --phase4 aug ssl      # 증강 + SSL

# 중단된 실험 이어서
python scripts/03_train/train.py --phase4 --resume

# 완료된 실험도 강제 재학습
python scripts/03_train/train.py --phase4 --force
```

#### 4-A. MUSAN 다운로드

```bash
# noise 카테고리만 (~1GB)
python scripts/01_download/download_musan.py -o data/musan --categories noise
```

#### 4-B. 테스트 시 노이즈 평가 (실험 A, 2×2 매트릭스)

기존 체크포인트로 평가만 수행. 재학습 없음. 8모델 × 6조건 × T1~T8.

```bash
# 전체 (8모델 × 6조건 × T1~T8, missing manifest 자동 skip)
python scripts/03_train/eval_noise.py --musan-dir data/musan/musan

# clean 학습 모델만
python scripts/03_train/eval_noise.py --models specrnet lcnn

# augmentation 모델만
python scripts/03_train/eval_noise.py --models specrnet_aug_musan lcnn_aug_both

# 임의 YAML config로 평가 (프리셋 외 모델)
python scripts/03_train/eval_noise.py --config scripts/03_train/configs/my_model.yaml

# 특정 조건/테스트셋만
python scripts/03_train/eval_noise.py --conditions clean gaussian_001 musan_10db codec_mp3_64
python scripts/03_train/eval_noise.py --eval-sets T1 T2 T4
```

**노이즈 조건 (6가지):**

| 조건 | 설명 | 비교 대상 |
|------|------|----------|
| clean | 원본 (대조군) | — |
| gaussian_001 | Gaussian α=0.001 (~50dB) | 임동훈 Table 3 |
| musan_20db | MUSAN SNR 20dB | 가벼운 잡음 |
| musan_10db | MUSAN SNR 10dB | 보통 잡음 |
| musan_5db | MUSAN SNR 5dB | 악조건 |
| codec_mp3_64 | MP3 64kbps 코덱 압축 | 실전 코덱 열화 |

**대상 모델 (8개 프리셋):**

| 그룹 | 모델 키 |
|------|--------|
| Clean 학습 | `specrnet`, `lcnn` |
| MUSAN 증강 | `specrnet_aug_musan`, `lcnn_aug_musan` |
| Codec 증강 | `specrnet_aug_codec`, `lcnn_aug_codec` |
| MUSAN+Codec | `specrnet_aug_both`, `lcnn_aug_both` |

#### 4-C. 학습 시 증강 (실험 B)

C_all 데이터에 증강을 적용하여 재학습. SpecRNet + LCNN 각각 수행.

```bash

# 사전 인코딩 (1회만 실행, C_all 전체)
python scripts/03_train/precompute_codec.py manifests/C_all/train.csv

# B-2: MUSAN 증강 (1순위)
python scripts/03_train/train.py --config scripts/03_train/configs/specrnet_call_aug_musan.yaml
python scripts/03_train/train.py --config scripts/03_train/configs/lcnn_call_aug_musan.yaml

# B-3: Codec 증강 (2순위)
python scripts/03_train/train.py --config scripts/03_train/configs/specrnet_call_aug_codec.yaml
python scripts/03_train/train.py --config scripts/03_train/configs/lcnn_call_aug_codec.yaml

# B-4: MUSAN + Codec (2순위)
python scripts/03_train/train.py --config scripts/03_train/configs/specrnet_call_aug_both.yaml
python scripts/03_train/train.py --config scripts/03_train/configs/lcnn_call_aug_both.yaml

# 또는 한번에: python scripts/03_train/train.py --phase4 aug
```

#### 4-D. 발화 길이별 성능 (실험 C, 2순위)

```bash
# C-1: 2초
python scripts/03_train/train.py --config scripts/03_train/configs/specrnet_call_2sec.yaml
python scripts/03_train/train.py --config scripts/03_train/configs/lcnn_call_2sec.yaml

# C-3: 8초
python scripts/03_train/train.py --config scripts/03_train/configs/specrnet_call_8sec.yaml
python scripts/03_train/train.py --config scripts/03_train/configs/lcnn_call_8sec.yaml

# 또는 한번에: python scripts/03_train/train.py --phase4 len
```

#### 4-E. SSL 베이스라인 (실험 D, 3순위)

XLS-R(300M frozen) + 경량 head로 SSL 베이스라인 성능 측정. `transformers` 패키지 필요.

```bash
# 사전 설치
pip install transformers>=4.40

# D-1: XLS-R → AASIST-style attention head (~296K trainable)
python scripts/03_train/train.py --config scripts/03_train/configs/xlsr_aasist_call.yaml

# D-2: XLS-R → FC (~2K trainable)
python scripts/03_train/train.py --config scripts/03_train/configs/xlsr_fc_call.yaml

# 또는 한번에: python scripts/03_train/train.py --phase4 ssl
```

### Phase 5: 평가 지표 보강

기존 13개 체크포인트를 재평가하여 minDCF, FPR/FNR을 추가 산출한다.

```bash
# 5-1. 전체 재평가 (minDCF + FPR/FNR 포함)
python scripts/03_train/build_results.py --all --output experiments/results_v2.csv

# 5-2. 임동훈(2024) 비교표 생성
python scripts/04_analyze/compare_lim.py --results experiments/results_v2.csv

# 5-3. 결과 분석 출력
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
| T4 | 노이즈 강건성표 (2×2 매트릭스) |

---

## 출력 디렉토리 구조

```
data/                               ← 원본 데이터
├── mlaad/fake/{lang}/{tts}/*.wav
├── mailabs/{locale}/{gender}/{speaker}/*.wav
├── speechfake/{MD,Real}/...
├── koaad/{tts_type}/*.wav
├── aihub/...
├── musan/musan/{noise,speech,music}/*.wav
├── asvspoof19/                         ← T4
├── asvspoof21/                         ← T5, T6
├── cdadd/                              ← T7
└── inthewild/                          ← T8

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
├── T3_mlaad_test.csv
├── T4_asvspoof19_la_eval.csv           ← 외부 벤치마크
├── T5_asvspoof21_df_eval.csv
├── T6_asvspoof21_la_eval.csv
├── T7_cdadd.csv
└── T8_inthewild.csv

experiments/                        ← 실험 결과
├── checkpoints/{experiment}_best.pt
├── logs/{experiment}.jsonl
├── tb_logs/{experiment}/
├── results.csv                         Phase 3 결과
├── results_v2.csv                      Phase 5 결과 (minDCF/FPR/FNR 포함)
├── noise_robustness.csv                Phase 4-A 노이즈 강건성
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
  name: specrnet                     # aasist | specrnet | lcnn | rawnet2 | xlsr_aasist | xlsr_fc

data:
  train_manifest: manifests/C_all/train.csv
  dev_manifest: manifests/C_all/dev.csv
  eval_manifests:
    T1: manifests/T1_speechfake_ko_test.csv
    T2: manifests/T2_koaad_holdout.csv
    T3: manifests/T3_mlaad_test.csv
    T4: manifests/T4_asvspoof19_la_eval.csv
    T5: manifests/T5_asvspoof21_df_eval.csv
    T6: manifests/T6_asvspoof21_la_eval.csv
    T7: manifests/T7_cdadd.csv
    T8: manifests/T8_inthewild.csv
  num_samples: 64000                 # 4초 × 16kHz (2초=32000, 8초=128000)
  num_workers: 4

training:
  optimizer: adam
  lr: 0.0001
  batch_size: 64                     # AASIST는 16, XLS-R은 8, 8초는 32
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

**설정 파일 목록 (25개):**

| Phase | 설정 파일 | 모델 | 데이터 | 비고 |
|-------|----------|------|--------|------|
| 3-1 | aasist_c1~c3, lcnn_c1~c3, rawnet2_c1~c3 | 3종 | C1/C2/C3 | 9개 |
| 3-2 | aasist_call, specrnet_call, lcnn_call, rawnet2_call | 4종 | C_all | 4개 |
| 4-B | specrnet_call_aug_musan, lcnn_call_aug_musan | 2종 | C_all+MUSAN | 2개 |
| 4-B | specrnet_call_aug_codec, lcnn_call_aug_codec | 2종 | C_all+Codec | 2개 |
| 4-B | specrnet_call_aug_both, lcnn_call_aug_both | 2종 | C_all+MUSAN+Codec | 2개 |
| 4-C | specrnet_call_2sec, lcnn_call_2sec | 2종 | C_all, 2초 | 2개 |
| 4-C | specrnet_call_8sec, lcnn_call_8sec | 2종 | C_all, 8초 | 2개 |
| 4-D | xlsr_aasist_call, xlsr_fc_call | 2종 | C_all, SSL | 2개 |

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
| ASVspoof 2019/2021 | CC BY 4.0 |
| InTheWild | CC BY-SA 4.0 |
| CD-ADD | 연구 목적 |
