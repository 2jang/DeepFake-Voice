# 연구 수행 절차 — Phase별 상세 가이드

**프로젝트**: 한국어 딥페이크 음성 탐지를 위한 경량 모델의 데이터 커버리지 효과와 실용적 강건성 분석
**총 기간**: 6주 (핵심 실험 완료 후 잔여 기간)
**수행된 실험**: 13개 (Phase 1: 9개, Phase 2: 4개) ✅
**남은 실험**: ~20개 (강건성 + 지표 보강 + SSL, SpecRNet+LCNN 병행)

---

## 진행 상태 요약

```
Phase 1: 데이터 수집 및 전처리        ✅ 완료
Phase 2: 모델 환경 구축               ✅ 완료
Phase 3: 핵심 실험 (13개)             ✅ 완료
Phase 4: 평가 지표 보강               ← 현재 단계
Phase 5: 강건성 실험 (~14개)          예정
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

### 1.4 평가셋 (전 실험 공통, 고정)

| 평가셋 | 내용 | 샘플 수 | 역할 |
|--------|------|--------|------|
| T1 | SpeechFake-MD ko test | 982 | 한국어 기본 평가 |
| T2 | KoAAD 홀드아웃 | 26,977 | 한국어 cross-dataset |
| T3 | MLAAD test (ko 제외) | 116,544 | 다국어 성능 유지 확인 |

### 1.5 완료 기준 ✅

- [x] 모든 파일 16kHz mono WAV
- [x] C1/C2/C3/C_all manifest CSV 생성
- [x] T1/T2/T3 평가셋 고정

---

## Phase 2: 모델 환경 구축 ✅ 완료

### 2.1 모델

| 모델 | 파라미터 | 입력 | 특징 |
|------|---------|------|------|
| AASIST | 229K | Raw waveform | Graph Attention + Spectro-Temporal |
| SpecRNet | 416K | 스펙트로그램 | ResNet + FMS Attention |
| LCNN | 1.37M | 스펙트로그램 | Light CNN + MaxFeatureMap |
| RawNet2 | 18.5M | Raw waveform | SincConv + GRU |

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

### 2.3 폴더 구조

```
scripts/03_train/
├── train.py              # 통합 학습/평가 스크립트
├── configs/              # 실험별 YAML
├── models/               # AASIST, LCNN, SpecRNet, RawNet2, XLS-R (SSL)
└── merge_manifests.py    # C_all manifest 생성

experiments/              # 결과 및 문서
manifests/                # C1, C2, C3, C_all CSV

D:\AI\_Syncthing Cloud\vastai1\  # 클라우드 결과
├── checkpoints/
├── logs/
├── tb_logs/
└── results.csv
```

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

## Phase 4: 평가 지표 보강 ← 현재 단계

기존 13개 실험에 대해 추가 평가 지표를 계산한다. 임동훈(2024)과의 직접 비교 및 ASVspoof 표준 지표 확보가 목적.

### 4.1 minDCF 계산

ASVspoof 및 임동훈(2024)이 사용한 표준 지표.

```
설정: C_miss=2, C_fa=1, P_target=0.5
대상: 13개 실험 전부 (T1, T2, T3 각각)
```

### 4.2 FPR/FNR 분리 분석

임동훈(2024) Table 5에서 한국어 FPR이 65~93%로 폭발한 문제의 해결 여부 검증.

```
대상: C_all 4개 모델의 T2(KoAAD) 결과
비교: 임동훈 Table 5 (LCNN-LSTM FPR 65.36%, Whisper-LCNN FPR 93.47%)
```

### 4.3 수행 절차

1. metrics.py에 minDCF, FPR/FNR(threshold=0.5), FPR/FNR@EER 계산 로직 추가 ✅
2. compute_all() 확장: 6개 → 11개 지표 ✅
3. test_metrics.py 단위 테스트 20개 작성 및 통과 ✅
4. 기존 13개 실험 체크포인트로 일괄 재평가 → results_v2.csv
5. results.csv에 minDCF, FPR, FNR, FPR@EER, FNR@EER 컬럼 추가

### 4.4 완료 기준

- [ ] minDCF 계산 코드 구현 및 검증
- [ ] 13개 실험 x 3개 테스트셋 = 39개 minDCF 값 산출
- [ ] C_all 4개 모델의 T2 FPR/FNR 산출
- [ ] 임동훈(2024) Table 4, 5와 대조표 작성

---

## Phase 5: 강건성 실험 (~14개)

RQ4(실전 환경 강건성)에 답하기 위한 실험. C_all 최적 모델(SpecRNet, LCNN) 중심 수행.

**일괄 실행**: `--phase5` 플래그로 증강(B) + 발화길이(C) + SSL(D) 12개 실험을 한 번에 실행 가능:
```bash
python scripts/03_train/train.py --phase5          # 12개 Phase 5 config 순차 실행
python scripts/03_train/train.py --phase5 --resume  # 중단된 실험 이어서
```

### 5.1 실험 A: 테스트 시 노이즈 강건성

임동훈(2024) Table 3과 직접 비교 가능한 설계. 학습 모델은 변경하지 않고, 테스트 데이터에만 노이즈를 추가한다. **SpecRNet + LCNN 두 모델 모두 평가** (재학습 없음, 기존 체크포인트 사용).

| # | 테스트 조건 | 모델 | 목적 | 우선순위 |
|---|-----------|------|------|---------|
| A-1 | Clean (대조군) | SpecRNet, LCNN | 기존 C_all 결과 재확인 | 필수 |
| A-2 | Gaussian noise α=0.001 | SpecRNet, LCNN | 임동훈 실험 조건 재현 | 필수 |
| A-3 | MUSAN 배경음 SNR 20dB | SpecRNet, LCNN | 가벼운 실환경 잡음 | 필수 |
| A-4 | MUSAN 배경음 SNR 10dB | SpecRNet, LCNN | 보통 실환경 잡음 | 필수 |
| A-5 | MUSAN 배경음 SNR 5dB | SpecRNet, LCNN | 악조건 잡음 | 필수 |

→ 5개 조건 × 2개 모델 = **10개 평가** (재학습 불필요)

**MUSAN 데이터 준비 필요**: noise, speech, music 카테고리 다운로드

수행 절차:
1. MUSAN 데이터셋 다운로드 및 전처리
2. 노이즈 추가 유틸리티 구현 (SNR 기반 혼합)
3. C_all 체크포인트(SpecRNet, LCNN) 두 모델로 각 조건 평가
4. 임동훈 Table 3과 비교표 작성 (LCNN 직접 비교)

### 5.2 실험 B: 학습 시 증강 효과

학습 과정에 증강을 적용하여 강건성 향상 여부를 확인. **SpecRNet + LCNN 두 모델 모두 재학습**.

| # | 학습 증강 | 모델 | 데이터 | 테스트 | 우선순위 |
|---|---------|------|--------|--------|---------|
| B-1 | 없음 (대조군) | SpecRNet, LCNN | C_all | A 전체 조건 | 기존 체크포인트 |
| B-2 | MUSAN noise SNR 10~20dB | SpecRNet, LCNN | C_all | A 전체 조건 | 필수 |
| B-3 | Codec 압축 (mp3 64kbps) | SpecRNet, LCNN | C_all | A 전체 조건 | 2순위 |
| B-4 | MUSAN + Codec 조합 | SpecRNet, LCNN | C_all | A 전체 조건 | 2순위 |

→ B-1은 기존 specrnet_call/lcnn_call 체크포인트 사용, B-2~B-4 × 2 모델 = **6개 신규 학습**

수행 절차:
1. AudioDataset에 on-the-fly 증강 옵션 추가
2. 각 증강 조건별 SpecRNet, LCNN + C_all 재학습 (조건당 2개 모델)
3. 실험 A의 5개 테스트 조건에서 두 모델 모두 평가
4. 증강 없음(B-1) 대비 개선폭 정량화, 모델 간 증강 효과 차이 분석

### 5.3 실험 C: 발화 길이별 성능

**SpecRNet + LCNN 두 모델 모두 재학습**.

| # | 입력 길이 | 모델 | 샘플 수 | 목적 | 우선순위 |
|---|----------|------|--------|------|---------|
| C-1 | 2초 | SpecRNet, LCNN | 32,000 | 극단 짧은 발화 | 2순위 |
| C-2 | 4초 | SpecRNet, LCNN | 64,000 | 현재 설정 (대조군) | 기존 체크포인트 |
| C-3 | 8초 | SpecRNet, LCNN | 128,000 | 황규한(2025) 최적 길이 | 2순위 |

→ C-2는 기존 체크포인트 사용, C-1/C-3 × 2 모델 = **4개 신규 학습**

수행 절차:
1. num_samples 파라미터 변경한 config 생성
2. SpecRNet, LCNN + C_all로 2초/8초 각각 재학습 (길이당 2개 모델)
3. 동일 테스트셋에서 두 모델 모두 평가, 4초 결과와 비교
4. 모델 간 길이 민감도 차이 분석 (스펙트로그램 vs 스펙트로그램)

### 5.4 실험 D: SSL 베이스라인 (코드 구현 완료)

RQ5(SSL 대비 격차) 답변용. 코드 구현 완료, 학습 실행 대기.

| # | 구성 | 학습 파라미터 | 목적 |
|---|------|-------------|------|
| D-1 | XLS-R (frozen) → AASIST head | ~296K | SSL vs 비SSL 격차 정량화 |
| D-2 | XLS-R (frozen) → FC | ~2K | SSL feature 순수 효과 |

수행 절차:
1. `pip install transformers` (XLS-R 모델 자동 다운로드)
2. `python scripts/03_train/train.py --phase5` 또는 개별 config 실행
3. 동일 테스트셋 평가, 비SSL 결과와 비교

구현 방식: XLS-R(wav2vec2-xls-r-300m, 300M frozen)을 on-the-fly feature extractor로 사용. 체크포인트는 head 파라미터만 저장.

### 5.5 완료 기준

- [ ] 실험 A (5조건 × 2모델 = 10개 평가): 테스트 시 노이즈 강건성 결과표
- [ ] 실험 B (3조건 × 2모델 = 6개 신규 학습 + 2개 대조군): 학습 시 증강 효과 비교표
- [ ] 실험 C (2길이 × 2모델 = 4개 신규 학습 + 2개 대조군): 발화 길이별 성능 비교
- [ ] 실험 D (2개): SSL 베이스라인 (코드 구현 완료, `--phase5`로 일괄 실행 가능)
- [ ] 임동훈(2024) Table 3과 직접 비교표 작성 (LCNN 중심)
- [ ] SpecRNet vs LCNN 강건성 차이 분석

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

### 6.2 필수 테이블 (논문용)

| # | 테이블 | 내용 |
|---|--------|------|
| T1 | 전체 결과표 | 13개 실험 x (EER, Accuracy, F1, AUC, minDCF) |
| T2 | C_all 모델 비교 | 4개 모델 x 성능 + 경량화 지표 |
| T3 | 임동훈 비교표 | 동일 모델, 다른 데이터 조건 EER/FPR 대조 |
| T4 | 노이즈 강건성표 | SNR별 EER 변화 |
| T5 | 증강 효과표 | 증강 전후 EER/minDCF 비교 |
| T6 | ASVspoof 간접 비교 | 참고용 (데이터셋 상이 명시) |

### 6.3 필수 그래프

| # | 그래프 | 유형 |
|---|--------|------|
| G1 | 데이터 커버리지별 EER | Grouped bar chart (모델 x 데이터 조합) |
| G2 | 성능-파라미터 트레이드오프 | 산점도 (x: params, y: EER, 크기: 추론속도) |
| G3 | SNR별 성능 하락 | Line chart (x: SNR, y: EER) |
| G4 | Phase 1 → Phase 2 개선 | Before/After bar chart |

### 6.4 완료 기준

- [ ] 분석 6개 항목 완료
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
   - 데이터 구성 (C1/C2/C3/C_all, T1/T2/T3)
   - 모델 4종 개요
   - 학습 설정
   - 평가 지표 (EER, minDCF, FPR/FNR, AUC)

4. 실험 결과
   4.1 데이터 커버리지 효과 (Phase 3-1 + 3-2)
   4.2 모델별 성능 비교 (C_all 기준)
   4.3 임동훈(2024) 직접 비교
   4.4 노이즈 강건성 (Phase 5)
   4.5 증강 효과 (Phase 5)
   4.6 발화 길이 (Phase 5, 여유 시)

5. 논의
   - 데이터 커버리지가 지배적 변인인 이유
   - SpecRNet 일반화 부족의 재평가
   - 파라미터-성능 비비례의 시사점
   - SSL 모델과의 격차 (여유 시)
   - 한계: 한국 상용 TTS 미검증, 모듈식 프레임워크 미적용

6. 결론
   - RQ1~5 답변 요약
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
Phase 4 (1주) ── minDCF, FPR/FNR 계산, 임동훈 비교표
     │
Phase 5 (2주) ── 노이즈 5조건×2모델 + 증강 3조건×2모델 + 발화길이 2×2모델 + SSL 2개
     │
Phase 6 (1주) ── 분석 6개, 테이블 6개, 그래프 4개
     │
Phase 7 (2주) ── 논문 집필, 교정
```

---

## 우선순위 정리

### 1순위 (필수 — 논문의 핵심)

- [ ] Phase 4: minDCF + FPR/FNR 계산 (임동훈 비교용)
- [ ] Phase 5 실험 A: 테스트 시 노이즈 5조건 × SpecRNet/LCNN (10개 평가)
- [ ] Phase 5 실험 B-2: MUSAN 증강 × SpecRNet/LCNN (2개 학습)
- [ ] Phase 6: 테이블 T1~T4, 그래프 G1~G4

### 2순위 (논문 보강)

- [ ] Phase 5 실험 B-3, B-4: Codec, MUSAN+Codec 증강 × SpecRNet/LCNN (4개 학습)
- [ ] Phase 5 실험 C: 발화 길이 2초/8초 × SpecRNet/LCNN (4개 학습)
- [ ] Phase 6: 테이블 T5 (증강 효과), SpecRNet vs LCNN 강건성 비교

### 3순위 (여유 시)

- [ ] Phase 5 실험 D: SSL 베이스라인 (코드 구현 완료)
- [ ] Phase 6: 테이블 T6 (ASVspoof 간접 비교)

---

*이 문서는 연구 진행에 따라 업데이트합니다.*
