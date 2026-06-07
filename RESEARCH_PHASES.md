# 연구 수행 절차 — Phase별 상세 가이드 + 코딩 계획

**프로젝트**: 한국어 딥페이크 음성 탐지를 위한 경량 모델의 데이터 커버리지 효과와 실용적 강건성 분석
**총 기간**: 6주 (핵심 실험 완료 후 잔여 기간)
**수행된 실험**: 13개 (Phase 1: 9개, Phase 2: 4개) ✅
**남은 실험**: ~20개 (강건성 + 지표 보강 + SSL, SpecRNet+LCNN 병행)

> 이 문서는 기존 `CODING_PLAN.md`를 통합하여 **연구 절차 + 코드 구현 계획**을 하나로 관리한다.

---

## 진행 상태 요약

```
Phase 1: 데이터 수집 및 전처리        ✅ 완료
Phase 2: 모델 환경 구축               ✅ 완료
Phase 3: 핵심 실험 (13개)             ✅ 완료
Phase 4: 강건성 실험 (~20개)          ← 현재 단계
Phase 5: 평가 지표 보강               예정
Phase 6: 분석 및 시각화               예정
Phase 7: 논문 작성                    예정
```

---

## Phase 1: 데이터 수집 및 전처리 ✅ 완료

### 1.1 데이터셋

| 데이터셋 | 출처 | 역할 |
|---------|------|------|
| MLAAD v9 | Fraunhofer ownCloud | 다국어 딥페이크 (51개 언어, 한국어 포함) |
| M-AILABS | caito.de | 다국어 실제 음성 (9개 언어) |
| SpeechFake-MD ko | GitHub (YMLLG/SpeechFake) | 한국어 딥페이크+실제 |
| KoAAD | GitHub (ldh-Hoon/KoAAD) | 한국어 딥페이크+실제 (소규모) |

### 1.2 전처리

```
원본 오디오
  → 리샘플링: 16kHz mono WAV 통일
  → 길이 정규화: 4초 (64,000 samples)
    - 4초 미만: zero-padding
    - 4초 초과: 랜덤 크롭 (학습) / 센터 크롭 (평가)
  → 품질 검증: 무음/클리핑/손상 파일 제거
```

### 1.3 데이터 조합

| 조합 | 구성 | 한국어 | 학습 샘플 수 | Dev 샘플 수 |
|------|------|--------|------------|------------|
| C1 | MLAAD v9 (ko 제외) + M-AILABS | X | 373,090 | 69,844 |
| C2 | MLAAD v9 (전체) + M-AILABS + SpeechFake ko | O | 555,802 | 95,800 |
| C3 | SpeechFake ko + KoAAD | O (Only) | 139,068 | 11,148 |
| C_all | C1 + C2 + C3 통합 | O | 1,067,960 | 176,792 |

### 1.4 평가셋

**내부 테스트셋 (전 실험 공통, 고정):**

| 평가셋 | 내용 | 샘플 수 | 역할 |
|--------|------|--------|------|
| T1 | SpeechFake-MD ko test | 982 | 한국어 기본 평가 |
| T2 | KoAAD 홀드아웃 | 26,977 | 한국어 cross-dataset |
| T3 | MLAAD test (ko 제외) | 116,544 | 다국어 성능 유지 확인 |

**외부 벤치마크 (SOTA 동일 조건 비교용):**

| 평가셋 | 내용 | 언어 | 규모 | 역할 |
|--------|------|------|------|------|
| T4 | ASVspoof 2019 LA eval | 영어 | 71,237 | SOTA 직접 비교 (AASIST 0.83%) |
| T5 | ASVspoof 2021 DF eval | 영어 | 593,253 | 코덱 열화 + 최신 공격 |
| T6 | ASVspoof 2021 LA eval | 영어 | — | 전화/VoIP 채널 효과 |
| T7 | CD-ADD | 중국어 | 300h+ | Cross-domain zero-shot TTS |
| T8 | InTheWild | 영어 | 37.9h | 실환경 유명인 딥페이크 |

### 1.5 완료 기준 ✅

- [x] 모든 파일 16kHz mono WAV
- [x] C1/C2/C3/C_all manifest CSV 생성
- [x] T1/T2/T3 평가셋 고정
- [x] T4~T8 외부 벤치마크 다운로드 스크립트 및 manifest 빌더 작성

---

## Phase 2: 모델 환경 구축 ✅ 완료

### 2.1 모델

| 모델 | 파라미터 | 입력 | 특징 |
|------|---------|------|------|
| AASIST | 229K | Raw waveform | Graph Attention + Spectro-Temporal |
| SpecRNet | 416K | 스펙트로그램 | ResNet + FMS Attention |
| LCNN | 1.37M | 스펙트로그램 | Light CNN + MaxFeatureMap |
| RawNet2 | 18.5M | Raw waveform | SincConv + GRU |

**SSL 베이스라인 (코드 구현 완료):**

| 모델 | 학습 파라미터 | 총 파라미터 | 특징 |
|------|-------------|-----------|------|
| XLSRAasist | ~296K | 300M+ (frozen) | XLS-R → attention-pooled head |
| XLSRfc | ~2K | 300M+ (frozen) | XLS-R → mean-pool + FC |

### 2.2 공통 학습 설정

```yaml
optimizer: Adam
lr: 1e-4
weight_decay: 1e-4
batch_size: 16 (AASIST) / 64 (나머지)
epochs: 50
early_stopping: patience 10 (Dev EER 기준)
seed: 42
input: 4초 (64,000 samples), 16kHz mono
loss: CrossEntropyLoss (2-class, class_weights: [1.0, 1.0] = 미가중)
GPU: RTX 4090 (클라우드)
최적화: torch.compile, bf16, tf32
```

### 2.3 코드 구조

```
scripts/03_train/
├── train.py              # 통합 학습/평가 스크립트
├── build_results.py      # 체크포인트 재평가 → results_v2.csv
├── eval_noise.py         # 테스트 시 노이즈 강건성 평가 (2×2 매트릭스)
├── dataset.py            # AudioDataset (증강 지원)
├── metrics.py            # EER, minDCF, FPR/FNR, AUC 등 11개 지표
├── test_metrics.py       # metrics.py 단위 테스트 (20개)
├── augment.py            # MUSANNoisePool, Gaussian, Codec 증강
├── sota.py               # ASVspoof 2019 SOTA 레퍼런스
├── merge_manifests.py    # C_all manifest 병합
├── verify_models.py      # 모델 forward/backward 검증
├── launch_tb.py          # TensorBoard 런처
├── configs/              # 25개 YAML
└── models/               # AASIST, LCNN, SpecRNet, RawNet2, XLS-R

scripts/01_download/
├── download_mlaad.py         # MLAAD v9
├── download_mailabs.py       # M-AILABS
├── download_speechfake.py    # SpeechFake-MD
├── download_koaad.py         # KoAAD
├── download_musan.py         # MUSAN (노이즈 증강용)
├── download_asvspoof19.py    # ASVspoof 2019 LA (T4)
├── download_asvspoof21.py    # ASVspoof 2021 DF+LA (T5, T6)
├── download_cdadd.py         # CD-ADD (T7)
└── download_inthewild.py     # InTheWild (T8)

scripts/02_preprocess/
├── preprocess.py             # 오디오 전처리
├── build_splits.py           # train/dev/test 분할
└── prepare_benchmarks.py     # T4~T8 manifest 빌더 (통합)

scripts/04_analyze/
├── analyze.py                # 결과 분석 + SOTA 비교
├── compare_lim.py            # 임동훈(2024) 비교표
└── visualize.py              # 논문용 그래프 + 테이블

experiments/                  # 결과 및 문서
manifests/                    # C1~C_all + T1~T8 CSV
```

**핵심 인터페이스:**
- `AudioDataset(manifest_path, num_samples, is_train, seed, augment_config)` → `(wav_tensor, label)`
- `compute_all(labels, scores)` → `{eer, mindcf, accuracy, f1, precision, recall, auc, fpr, fnr, fpr_eer, fnr_eer}` (11개)
- YAML config: `experiment`, `model.name`, `data.*`, `training.*`, `augmentation` (선택)

### 2.4 완료 기준 ✅

- [x] 4개 모델 forward/backward 정상
- [x] 통합 학습/평가 스크립트 완성
- [x] TensorBoard 로깅 + 원격 접근
- [x] 학습 재개(resume) 기능

---

## Phase 3: 핵심 실험 ✅ 완료 (13개)

### 3.1 Phase 3-1: 데이터 조합별 비교 (9개)

3개 모델(AASIST, LCNN, RawNet2) x 3개 데이터 조합(C1, C2, C3)

| # | 실험 | 모델 | 데이터 | Dev EER | T1 EER | T2 EER | T3 EER |
|---|------|------|--------|---------|--------|--------|--------|
| 1 | aasist_c1 | AASIST | C1 | 9.36% | 71.60% | 28.75% | 3.59% |
| 2 | aasist_c2 | AASIST | C2 | 0.71% | 1.43% | 35.29% | 0.52% |
| 3 | aasist_c3 | AASIST | C3 | 30.86% | 9.98% | 1.42% | 37.78% |
| 4 | lcnn_c1 | LCNN | C1 | 12.27% | 62.95% | 76.37% | 0.00% |
| 5 | lcnn_c2 | LCNN | C2 | 0.01% | 0.20% | 57.35% | 0.01% |
| 6 | lcnn_c3 | LCNN | C3 | 0.25% | 0.20% | 0.43% | 81.30% |
| 7 | rawnet2_c1 | RawNet2 | C1 | 12.64% | 66.01% | 38.59% | 0.44% |
| 8 | rawnet2_c2 | RawNet2 | C2 | 1.67% | 0.81% | 59.79% | 1.72% |
| 9 | rawnet2_c3 | RawNet2 | C3 | 29.53% | 10.18% | 0.47% | 65.81% |

**핵심 발견**: 모든 모델에서 학습 데이터에 미포함된 도메인의 테스트셋 실패 (EER 28~81%)

### 3.2 Phase 3-2: C_all 통합 데이터 (4개)

4개 모델 x C_all

| # | 실험 | 모델 | Params | Dev EER | T1 EER | T2 EER | T3 EER | 추론 | 처리량 |
|---|------|------|--------|---------|--------|--------|--------|------|--------|
| 10 | aasist_call | AASIST | 229K | 0.60% | 2.44% | 1.50% | 0.57% | 2.3ms | 431 sps |
| 11 | specrnet_call | SpecRNet | 416K | 0.07% | 0.20% | 0.16% | 0.05% | 1.4ms | 728 sps |
| 12 | lcnn_call | LCNN | 1.37M | 0.03% | 0.20% | 0.19% | 0.02% | 0.7ms | 1,360 sps |
| 13 | rawnet2_call | RawNet2 | 18.5M | 0.10% | 0.61% | 0.37% | 0.21% | 4.3ms | 235 sps |

**핵심 발견**: C_all이 모든 테스트셋에서 일반화 실패를 해결. SpecRNet이 효율 최적 모델.

### 3.3 완료 기준 ✅

- [x] 13개 실험 완료
- [x] results.csv 작성
- [x] 체크포인트 저장
- [x] RQ1~3 답변 확보

---

## Phase 4: 강건성 실험 (~20개) ← 현재 단계

RQ4(실전 환경 강건성), RQ5(SSL 트레이드오프), RQ6(외부 벤치마크 일반화)에 답하기 위한 실험. C_all 최적 모델(SpecRNet, LCNN) 중심 수행.

**일괄 실행**: `--phase4` 플래그로 증강(B) + 발화길이(C) + SSL(D) 12개 실험을 한 번에 실행 가능:
```bash
# Phase 4 전체 (증강 6 + 길이 4 + SSL 2 = 12개 학습)
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

### 4.1 실험 A: 테스트 시 노이즈 강건성 (eval_noise.py)

학습×평가 2×2 매트릭스 설계. 기존 체크포인트로 평가만 수행 (재학습 없음).

| 학습 조건 | Clean 평가 | Noise 평가 |
|---|---|---|
| **Clean 학습** | Phase 3 결과 (완료) | eval_noise.py |
| **Noise 학습** | Phase 4-B 학습 후 자동 평가 | eval_noise.py |

**노이즈 조건 (6가지):**

| 조건 | 설명 | 비교 대상 |
|------|------|----------|
| clean | 원본 (대조군) | — |
| gaussian_001 | Gaussian α=0.001 (~50dB SNR) | 임동훈 Table 3 |
| musan_20db | MUSAN SNR 20dB | 가벼운 잡음 |
| musan_10db | MUSAN SNR 10dB | 보통 잡음 |
| musan_5db | MUSAN SNR 5dB | 악조건 |
| codec_mp3_64 | MP3 64kbps 코덱 압축 | 실전 코덱 열화 |

**대상 모델 (8개 프리셋):**

| 그룹 | 모델 | Config |
|------|------|--------|
| Clean | specrnet, lcnn | specrnet_call.yaml, lcnn_call.yaml |
| MUSAN 증강 | specrnet_aug_musan, lcnn_aug_musan | 증강 학습 후 |
| Codec 증강 | specrnet_aug_codec, lcnn_aug_codec | 증강 학습 후 |
| MUSAN+Codec | specrnet_aug_both, lcnn_aug_both | 증강 학습 후 |

→ 8모델 × 6조건 × 8테스트셋(T1~T8) = 최대 **384개 평가** (missing manifest 자동 skip)

**실행:**
```bash
# 전체 (8모델 × 6조건, T1~T8)
python scripts/03_train/eval_noise.py --musan-dir data/musan/musan

# clean 학습 모델만
python scripts/03_train/eval_noise.py --models specrnet lcnn

# augmentation 모델만
python scripts/03_train/eval_noise.py --models specrnet_aug_musan specrnet_aug_both

# 임의 config로 평가
python scripts/03_train/eval_noise.py --config scripts/03_train/configs/my_model.yaml

# 특정 조건/테스트셋만
python scripts/03_train/eval_noise.py --conditions clean musan_10db codec_mp3_64
python scripts/03_train/eval_noise.py --eval-sets T1 T2 T4
```

**코드 구현 (eval_noise.py):**
- `NoisyAudioDataset`: AudioDataset 래퍼, `sr` param 지원, `apply_codec` 연동
- 인덱스별 결정적 RNG (`default_rng(seed+idx)`) → 모델 간 동일 노이즈 입력 보장
- `--config` (mutually exclusive with `--models`): 임의 YAML 경로로 평가
- 결과: `experiments/noise_robustness.csv`

### 4.2 실험 B: 학습 시 증강 효과

C_all 데이터에 증강을 적용하여 재학습. SpecRNet + LCNN 각각 수행.

| # | 학습 증강 | Config (SpecRNet) | Config (LCNN) |
|---|---------|-------------------|---------------|
| B-1 | 없음 (대조군) | specrnet_call | lcnn_call |
| B-2 | MUSAN SNR 10~20dB | specrnet_call_aug_musan | lcnn_call_aug_musan |
| B-3 | Codec MP3 64kbps | specrnet_call_aug_codec | lcnn_call_aug_codec |
| B-4 | MUSAN + Codec 조합 | specrnet_call_aug_both | lcnn_call_aug_both |

→ B-1은 기존 체크포인트, B-2~B-4 × 2모델 = **6개 신규 학습**

**코드 구현:**
- `augment.py`: `MUSANNoisePool`, `add_gaussian_noise`, `add_noise_snr`, `apply_codec` (PyAV in-memory)
- `dataset.py`: `augment_config` 파라미터 추가, `_apply_augmentation()` 메서드
- `train.py`: config의 `augmentation` 섹션 → AudioDataset 전달

### 4.3 실험 C: 발화 길이별 성능

SpecRNet + LCNN 두 모델 모두 재학습.

| # | 입력 길이 | 샘플 수 | Config (SpecRNet) | Config (LCNN) |
|---|----------|--------|-------------------|---------------|
| C-1 | 2초 | 32,000 | specrnet_call_2sec | lcnn_call_2sec |
| C-2 | 4초 (대조군) | 64,000 | specrnet_call | lcnn_call |
| C-3 | 8초 | 128,000 | specrnet_call_8sec | lcnn_call_8sec |

→ C-1/C-3 × 2모델 = **4개 신규 학습**

### 4.4 실험 D: SSL 베이스라인 (코드 구현 완료)

RQ5(SSL 대비 격차) 답변용. XLS-R(wav2vec2-xls-r-300m, 300M frozen) → 경량 head.

| # | 구성 | 학습 파라미터 | Config |
|---|------|-------------|--------|
| D-1 | XLS-R → AASIST head | ~296K | xlsr_aasist_call |
| D-2 | XLS-R → FC | ~2K | xlsr_fc_call |

**코드 구현 (xlsr_head.py):**
- `_XLSRBase`: frozen XLS-R + trainable head
- `state_dict` 오버라이드: head 파라미터만 저장/로드 (체크포인트 ~KB)
- `train()` 오버라이드: feature_extractor를 항상 eval 모드로 유지
- `torch.compile` 호환: `_orig_mod.` 접두사 자동 처리

### 4.5 외부 벤치마크 (T4~T8)

SOTA와 동일 데이터셋에서 비교하기 위한 외부 벤치마크.

**다운로드 스크립트:**

| 스크립트 | 대상 | 비고 |
|---------|------|------|
| `download_asvspoof19.py` | ASVspoof 2019 LA | Edinburgh DataShare, `--eval-only` |
| `download_asvspoof21.py` | ASVspoof 2021 DF+LA | Zenodo, eval keys 별도 |
| `download_cdadd.py` | CD-ADD | OpenXLab / Baidu Cloud |
| `download_inthewild.py` | InTheWild | HuggingFace (~8.16GB) |

**Manifest 빌더 (`prepare_benchmarks.py`):**
- 5개 데이터셋 통합 처리: protocol 파일 파싱, 16kHz 리샘플링, CSV 생성
- `--all` 옵션: 다운로드된 데이터셋만 자동 처리
- 출력: `manifests/T4_asvspoof19_la_eval.csv` ~ `manifests/T8_inthewild.csv`
- `train.py`는 missing manifest 자동 skip (`Path(eval_path).exists()` 체크)

### 4.6 완료 기준

- [ ] 실험 A: 2×2 매트릭스 노이즈 강건성 (8모델 × 6조건 × T1~T8)
- [ ] 실험 B: 학습 시 증강 (6개 신규 학습)
- [ ] 실험 C: 발화 길이 (4개 신규 학습)
- [ ] 실험 D: SSL 베이스라인 (2개, 코드 구현 완료)
- [ ] 외부 벤치마크 T4~T8 다운로드 및 평가
- [ ] 임동훈(2024) Table 3과 직접 비교표 (LCNN 중심)
- [ ] SpecRNet vs LCNN 강건성 차이 분석

### 4.7 YAML 설정 (Phase 4용 12개)

| 서브셋 | 설정 파일 | 모델 | 비고 |
|--------|----------|------|------|
| aug | specrnet_call_aug_musan, lcnn_call_aug_musan | 2종 | MUSAN 증강 |
| aug | specrnet_call_aug_codec, lcnn_call_aug_codec | 2종 | Codec 증강 |
| aug | specrnet_call_aug_both, lcnn_call_aug_both | 2종 | MUSAN+Codec |
| len | specrnet_call_2sec, lcnn_call_2sec | 2종 | 2초 |
| len | specrnet_call_8sec, lcnn_call_8sec | 2종 | 8초 |
| ssl | xlsr_aasist_call, xlsr_fc_call | 2종 | SSL 베이스라인 |

---

## Phase 5: 평가 지표 보강

기존 13개 실험에 대해 추가 평가 지표를 계산한다. 임동훈(2024)과의 직접 비교 및 ASVspoof 표준 지표 확보가 목적.

### 5.1 minDCF 계산

ASVspoof 및 임동훈(2024)이 사용한 표준 지표.

```
설정: C_miss=2, C_fa=1, P_target=0.5
대상: 13개 실험 전부 (T1, T2, T3 각각)
```

**코드 구현 (metrics.py):**
```python
def compute_mindcf(labels, scores, c_miss=2.0, c_fa=1.0, p_target=0.5) -> float:
    """DCF(θ) = c_miss * p_target * FNR(θ) + c_fa * (1-p_target) * FPR(θ)"""
```

### 5.2 FPR/FNR 분리 분석

임동훈(2024) Table 5에서 한국어 FPR이 65~93%로 폭발한 문제의 해결 여부 검증.

```
대상: C_all 4개 모델의 T2(KoAAD) 결과
비교: 임동훈 Table 5 (LCNN-LSTM FPR 65.36%, Whisper-LCNN FPR 93.47%)
```

**코드 구현 (metrics.py):**
- `compute_fpr_fnr(labels, scores, threshold=0.5)`: 실용적 배포 기준
- `compute_fpr_fnr_at_eer(labels, scores)`: EER threshold 기준 (학술 비교)
- `compute_all()` 확장: 6개 → 11개 지표 ✅

### 5.3 수행 절차

1. metrics.py에 minDCF, FPR/FNR, FPR/FNR@EER 추가 ✅
2. compute_all() 확장 (11개 지표) ✅
3. test_metrics.py 단위 테스트 20개 작성 및 통과 ✅
4. 기존 13개 실험 체크포인트로 일괄 재평가 → results_v2.csv
5. 임동훈(2024) 비교표 생성 (compare_lim.py)

**실행:**
```bash
# 일괄 재평가
python scripts/03_train/build_results.py --all --output experiments/results_v2.csv

# 임동훈 비교표
python scripts/04_analyze/compare_lim.py --results experiments/results_v2.csv
```

### 5.4 완료 기준

- [x] minDCF 계산 코드 구현 및 검증
- [ ] 13개 실험 x 3개 테스트셋 = 39개 minDCF 값 산출
- [ ] C_all 4개 모델의 T2 FPR/FNR 산출
- [ ] 임동훈(2024) Table 4, 5와 대조표 작성

---

## Phase 6: 분석 및 시각화

### 6.1 필수 분석 항목

| # | 분석 | 방법 | 답하는 질문 |
|---|------|------|-----------|
| 1 | 데이터 커버리지 ablation | C1/C2/C3/C_all EER 비교 바 차트 | RQ2 |
| 2 | 모델 효율성 비교 | 파라미터 vs EER 산점도, 추론 속도 포함 | RQ3 |
| 3 | 임동훈(2024) 직접 비교 | KoAAD 성능 대조표 + FPR/FNR 비교 | RQ1 |
| 4 | 노이즈 성능 하락 곡선 | SNR별 EER 변화 라인 차트 | RQ4 |
| 5 | 증강 효과 | 증강 유/무 비교표 | RQ4 |
| 6 | 성능-비용 트레이드오프 | EER vs 파라미터/추론속도/FLOPs | RQ3, RQ5 |
| 7 | 외부 벤치마크 성능 | T4~T8에서의 SOTA 동일 조건 비교 | RQ6 |

### 6.2 필수 테이블 (논문용)

| # | 테이블 | 내용 |
|---|--------|------|
| T1 | 전체 결과표 | 13개 실험 x (EER, Accuracy, F1, AUC, minDCF) |
| T2 | C_all 모델 비교 | 4개 모델 x 성능 + 경량화 지표 |
| T3 | 임동훈 비교표 | 동일 모델, 다른 데이터 조건 EER/FPR 대조 |
| T4 | 노이즈 강건성표 | 조건별 EER 변화 (2×2 매트릭스) |
| T5 | 증강 효과표 | 증강 전후 EER/minDCF 비교 |
| T6 | 외부 벤치마크 비교 | T4~T8에서 SOTA 대비 경량 모델 성능 |

### 6.3 필수 그래프

| # | 그래프 | 유형 |
|---|--------|------|
| G1 | 데이터 커버리지별 EER | Grouped bar chart (모델 x 데이터 조합) |
| G2 | 성능-파라미터 트레이드오프 | 산점도 (x: params, y: EER, 크기: 추론속도) |
| G3 | SNR별 성능 하락 | Line chart (x: SNR, y: EER) |
| G4 | Phase 1 → Phase 2 개선 | Before/After bar chart |

**코드 구현 (visualize.py):**
```bash
python scripts/04_analyze/visualize.py --all --results experiments/results_v2.csv
python scripts/04_analyze/visualize.py --graph G1
python scripts/04_analyze/visualize.py --tables
```

### 6.4 완료 기준

- [ ] 분석 7개 항목 완료
- [ ] 논문용 테이블 6개 완성
- [ ] 논문용 그래프 4개 완성

---

## Phase 7: 논문 작성

### 7.1 논문 구조

```
1. 서론
   - 딥페이크 음성 위협, 한국어 탐지 공백
   - 임동훈(2024) 50% 실패 문제 제기
   - 본 연구의 접근: 한국어 포함 학습 + 데이터 커버리지 분석

2. 관련 연구
   - 임동훈(2024): LCNN 개선, 한국어 실패 보고
   - SpeechFake(2025): 다국어 데이터셋, cross-lingual 전이
   - 황규한(2025): 발화 길이와 증강 효과
   - SpecRNet, AASIST, RawNet2 원 논문

3. 방법
   - 데이터 구성 (C1/C2/C3/C_all, T1~T8)
   - 모델 4종 + SSL 2종 개요
   - 학습 설정
   - 평가 지표 (EER, minDCF, FPR/FNR, AUC)

4. 실험 결과
   4.1 데이터 커버리지 효과 (Phase 3-1 + 3-2)
   4.2 모델별 성능 비교 (C_all 기준)
   4.3 임동훈(2024) 직접 비교
   4.4 노이즈 강건성 (Phase 4-A, 2×2 매트릭스)
   4.5 증강 효과 (Phase 4-B)
   4.6 발화 길이 (Phase 4-C)
   4.7 외부 벤치마크 (T4~T8, Phase 4)
   4.8 SSL 비교 (Phase 4-D)

5. 논의
   - 데이터 커버리지가 지배적 변인인 이유
   - SpecRNet 일반화 부족의 재평가
   - 파라미터-성능 비비례의 시사점
   - SSL 모델과의 격차
   - 한계: 한국 상용 TTS 미검증

6. 결론
   - RQ1~6 답변 요약
   - 실용적 권장 사항
   - 후속 연구 방향
```

### 7.2 산출물

- [ ] 논문 원고 (한국어 학회 논문 형식)
- [ ] 발표 슬라이드 (필요 시)

---

## 전체 일정

```
Phase 1~3 (완료) ── 데이터 4개 준비, 모델 4개 세팅, 13개 실험 완료
     │
Phase 4 (2주) ── 노이즈 6조건×8모델 + 증강 6학습 + 길이 4학습 + SSL 2 + 외부 벤치마크
     │
Phase 5 (1주) ── minDCF, FPR/FNR 계산, 임동훈 비교표
     │
Phase 6 (1주) ── 분석 7개, 테이블 6개, 그래프 4개
     │
Phase 7 (2주) ── 논문 집필, 교정
```

---

## 우선순위 정리

### 1순위 (필수 — 논문의 핵심)

- [ ] Phase 4 실험 A: 테스트 시 노이즈 6조건 × 8모델 (eval_noise.py)
- [ ] Phase 4 실험 B-2: MUSAN 증강 × SpecRNet/LCNN (2개 학습)
- [ ] Phase 5: minDCF + FPR/FNR 계산 (임동훈 비교용)
- [ ] Phase 6: 테이블 T1~T4, 그래프 G1~G4

### 2순위 (논문 보강)

- [ ] Phase 4 실험 B-3, B-4: Codec, MUSAN+Codec 증강 (4개 학습)
- [ ] Phase 4 실험 C: 발화 길이 2초/8초 (4개 학습)
- [ ] Phase 4: 외부 벤치마크 T4~T8 다운로드 + 평가
- [ ] Phase 6: 테이블 T5~T6

### 3순위 (여유 시)

- [ ] Phase 4 실험 D: SSL 베이스라인 (코드 구현 완료, `--phase4 ssl`)

---

## 코드 변경 이력

### 파일 변경 요약

| 파일 | 변경 유형 | 상태 | 내용 |
|------|----------|:----:|------|
| `scripts/03_train/metrics.py` | **수정** | ✅ | minDCF, FPR/FNR, FPR/FNR@EER, compute_all 11개 |
| `scripts/03_train/dataset.py` | **수정** | ✅ | augment_config, _apply_augmentation |
| `scripts/03_train/train.py` | **수정** | ✅ | augmentation 연동, `--phase4` 플래그, `--resume`/`--force` |
| `scripts/03_train/augment.py` | **신규** | ✅ | MUSANNoisePool, Gaussian, MUSAN SNR, Codec (PyAV) |
| `scripts/03_train/eval_noise.py` | **신규** | ✅ | 2×2 매트릭스 노이즈 평가, 8모델×6조건 |
| `scripts/03_train/test_metrics.py` | **신규** | ✅ | 단위 테스트 20개 |
| `scripts/03_train/models/xlsr_head.py` | **신규** | ✅ | XLSRAasist, XLSRfc |
| `scripts/04_analyze/compare_lim.py` | **신규** | ✅ | 임동훈 비교표 |
| `scripts/04_analyze/visualize.py` | **신규** | ✅ | G1~G4 + T1~T4 |
| `scripts/01_download/download_asvspoof19.py` | **신규** | ✅ | ASVspoof 2019 LA |
| `scripts/01_download/download_asvspoof21.py` | **신규** | ✅ | ASVspoof 2021 DF+LA |
| `scripts/01_download/download_cdadd.py` | **신규** | ✅ | CD-ADD |
| `scripts/01_download/download_inthewild.py` | **신규** | ✅ | InTheWild |
| `scripts/02_preprocess/prepare_benchmarks.py` | **신규** | ✅ | T4~T8 manifest 빌더 |
| configs/ (25개 YAML) | **수정/신규** | ✅ | T4~T8 eval_manifests 추가 |

### 코드 검토 수정 이력

| # | 심각도 | 문제 | 수정 |
|---|--------|------|------|
| 1 | 🔴 | `torch.compile` 체크포인트 `_orig_mod.` 접두사 → 로드 실패 | `load_checkpoint()` 헬퍼 |
| 2 | 🟠 | eval_noise 비결정적 노이즈 | `default_rng(seed+idx)` |
| 3 | 🟠 | codec ffmpeg subprocess 병목 | PyAV in-memory 재작성 |
| 4 | 🟡 | `--all` 완료 실험 재학습 | `--force` 분리 |
| 5 | 🟡 | build_results.py 기본 출력 충돌 | `results_v2.csv`로 변경 |
| 6 | 🟡 | "Weighted BCE" 표기 오류 | CrossEntropyLoss 정정 |

---

## GPU 실행 시간 추정

| 단계 | 내용 | GPU 시간 |
|------|------|---------|
| Phase 4-A | 8모델 × 6조건 × T1~T8 평가 (학습 없음) | ~2시간 |
| Phase 4-B | 6개 신규 학습 (C_all 규모) | ~12~18시간 |
| Phase 4-C | 4개 신규 학습 | ~8~12시간 |
| Phase 4-D | 2개 SSL (선택) | ~4~6시간 |
| Phase 5 | 13개 체크포인트 재평가 | ~30분 |
| **합계** | | **~27~39시간** |

---

*이 문서는 연구 진행에 따라 업데이트합니다.*
