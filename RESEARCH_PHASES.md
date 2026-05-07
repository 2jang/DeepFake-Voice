# 연구 수행 절차 — Phase별 상세 가이드

**프로젝트**: 한국어 딥페이크 음성 탐지를 위한 경량 탐지 모델의 성능 한계 실증 분석  
**총 기간**: 8주  
**총 실험 규모**: 216회 학습·평가, 1,512개 데이터 포인트

---

## Phase 1: 데이터 수집 및 전처리 (1주차)

### 1.1 데이터셋 다운로드

| 데이터셋 | 출처 | 용도 |
|----------|------|------|
| MLAAD v9 | Fraunhofer ownCloud | 다국어 딥페이크 (51개 언어, 한국어 포함) |
| M-AILABS | caito.de | 다국어 실제 음성 (낭독) |
| SpeechFake-MD ko | GitHub (YMLLG/SpeechFake) | 한국어 딥페이크+실제 |
| KoAAD | GitHub (ldh-Hoon/KoAAD) | 한국어 딥페이크+실제 (소규모) |
| MUSAN | OpenSLR | 증강용 잡음/음악/환경음 |

### 1.2 전처리 파이프라인

```
원본 오디오
  → 리샘플링: 16kHz mono WAV 통일
  → 길이 정규화: 4초 기준 (64,000 samples)
    - 4초 미만: zero-padding
    - 4초 초과: 랜덤 크롭 (학습) / 센터 크롭 (평가)
  → 품질 검증: 무음 비율, 클리핑, 손상 파일 제거
  → 메타데이터 생성: 파일명, 언어, 라벨(real/fake), 화자ID, TTS종류
```

### 1.3 데이터 조합별 분할

| 조합 | 학습 데이터 구성 | 분할 방식 |
|------|-----------------|-----------|
| C1 | MLAAD v9 (`lang != 'ko'` 필터링) + M-AILABS | Speaker-disjoint 8:1:1 |
| C2 | MLAAD v9 (전체) + M-AILABS + SpeechFake-MD ko + KoAAD | Speaker-disjoint 8:1:1 |
| C3 | SpeechFake-MD ko + KoAAD | Speaker-disjoint 8:1:1 |

### 1.4 평가셋 고정

| 평가셋 | 내용 | 비고 |
|--------|------|------|
| T1 | SpeechFake-MD ko test split | 한국어 기본 평가 |
| T2 | KoAAD 홀드아웃 | 한국어 cross-dataset |
| T3 | MLAAD test split | 다국어 성능 유지 확인 |

### 1.5 검증 기준

- [ ] 모든 파일 16kHz mono WAV 변환 완료
- [ ] Speaker-disjoint 분할 검증 (train/dev/test 화자 비중복)
- [ ] Class balance 확인 (bonafide:spoof ≈ 1:1)
- [ ] C1/C2/C3 manifest CSV 생성 완료
- [ ] T1/T2/T3 평가셋 고정 및 별도 저장

---

## Phase 2: 베이스라인 재현 (2주차)

### 2.1 목표

임동훈(2024) 결과를 재현하여 실험 환경의 신뢰성 확보.

### 2.2 재현 대상

| 모델 | 원논문 한국어 EER | 허용 오차 |
|------|------------------|-----------|
| LCNN-LSTM | 50.85% | ±2%p |
| Whisper-LCNN | 66.36% | ±2%p |
| SpecRNet | 57.42% | ±2%p |

### 2.3 수행 절차

1. AASIST 공식 코드(clovaai/aasist) 클론 및 환경 구축
2. C1 데이터(MLAAD ko 제외 + M-AILABS)로 LCNN-LSTM 학습
3. KoAAD(T2)로 평가 → 임동훈 결과와 비교
4. ±2%p 이내 확인 시 환경 검증 완료

### 2.4 학습 하이퍼파라미터 (공통)

```yaml
optimizer: Adam
learning_rate: 0.0001
scheduler: CosineAnnealingLR
batch_size: 32
epochs: 50
seed: 42
loss: Weighted BCE (0.9 fake / 0.1 real) 또는 1:1 BCE
early_stopping: patience 10 (dev EER 기준)
```

### 2.5 검증 기준

- [ ] LCNN-LSTM C1→T2 EER이 임동훈(2024) 대비 ±2%p 이내
- [ ] 학습 곡선(loss, EER) 정상 수렴 확인
- [ ] 실험 로그 및 체크포인트 저장 체계 구축

---

## Phase 3: 모듈 구현 및 단위 테스트 (3주차)

### 3.1 모듈 구현 목록

#### Stage 1: Feature Extractor (3종)

| 모듈 | 입력 | 출력 | 핵심 설정 |
|------|------|------|-----------|
| LFCC+ΔΔΔ | raw waveform | (T, F) | n_filters=20, with delta+double-delta |
| Mel-Spectrogram | raw waveform | (T, F) | n_mels=80, n_fft=512, hop=160 |
| Raw Waveform | raw waveform | (1, L) | 그대로 전달 (end-to-end) |

#### Stage 2: Enhancement (3종)

| 모듈 | 기능 | 출처 |
|------|------|------|
| HPF Block | 고주파 강조 필터 | 임동훈(2024) |
| Band-Attention | 주파수 대역별 가중치 학습 | 황규한(2025) LF Mask 기반 |
| None | 패스스루 (대조군) | — |

#### Stage 3: Backbone (4종)

| 모듈 | 파라미터 | 특징 |
|------|---------|------|
| AASIST | 297K | Graph Attention, spectro-temporal |
| LCNN | 1.5M | Light CNN, MaxFeatureMap |
| SpecRNet | 0.9M | 경량 spectrogram 분류 |
| RawNet2 | 17M | Raw waveform end-to-end |

#### Stage 4: Aggregation (2종)

| 모듈 | 기능 |
|------|------|
| BiLSTM | 시퀀스 양방향 집약 |
| Attention Pooling | 가변 길이 가중 평균 |

### 3.2 모듈 호환성 매트릭스

```
Feature       → Enhancement → Backbone   → Aggregation
─────────────────────────────────────────────────────────
LFCC+ΔΔΔ     → HPF/Band/None → AASIST    → BiLSTM/AttPool
Mel-Spec      → HPF/Band/None → LCNN      → BiLSTM/AttPool
Raw Waveform  → HPF/Band/None → SpecRNet  → BiLSTM/AttPool
                               → RawNet2   → BiLSTM/AttPool
```

**주의**: Raw Waveform은 RawNet2와 자연스럽게 결합. 스펙트럼 기반 Backbone(AASIST, LCNN, SpecRNet)은 LFCC/Mel-Spec 입력 필요. 호환 불가 조합은 어댑터 레이어 삽입 또는 실험에서 제외 후 문서화.

### 3.3 단위 테스트

각 모듈별:
1. 입출력 shape 검증 (dummy tensor)
2. forward/backward pass 정상 동작
3. 메모리 누수 없음 확인
4. 72개 조합 중 무작위 5개 end-to-end 동작 확인

### 3.4 검증 기준

- [ ] 4개 Stage 전체 모듈 구현 완료
- [ ] 72개 조합 모두 forward pass 가능 (또는 호환 불가 조합 명시)
- [ ] config 기반 조합 생성 자동화 (YAML/JSON)
- [ ] 단위 테스트 전수 통과

---

## Phase 4: 실험 A — 아키텍처 × 데이터 그리드 탐색 (4주차)

### 4.1 실험 규모

| 항목 | 값 |
|------|---|
| 아키텍처 조합 | 72개 |
| 데이터 조합 | 3개 (C1, C2, C3) |
| 총 학습 횟수 | 216회 |
| 증강 | 없음 (순수 아키텍처+데이터 효과) |
| 평가셋 | T1, T2, T3 |

### 4.2 실행 전략 (우선순위)

```
Priority 1: AASIST(297K) × 전체 Feature/Enhance/Agg × C1/C2/C3
            → 36회, 예상 ~72시간
Priority 2: SpecRNet(0.9M) × 전체 × C1/C2/C3
            → 36회, 예상 ~108시간
Priority 3: LCNN(1.5M) × 전체 × C1/C2/C3
            → 36회, 예상 ~144시간
Priority 4: RawNet2(17M) × 전체 × C1/C2/C3
            → 36회, 예상 ~288시간 (시간 부족 시 축소)
```

### 4.3 경량 지표 측정 (아키텍처별 1회)

72개 아키텍처 각각에 대해:
- E1: Inference Time (ms) — 100회 평균
- E2: Model Size (MB)
- E3: FLOPs (MFLOPs)
- E4: Peak Memory (MB)
- E5: Throughput (samples/sec)

### 4.4 로깅 및 저장

```
experiments/
├── exp_A/
│   ├── configs/          # 72개 조합 YAML
│   ├── logs/             # TensorBoard / WandB
│   ├── checkpoints/      # best model per run
│   └── results/
│       ├── accuracy_f1.csv     # 216행 × (조합정보 + Acc + F1)
│       └── efficiency.csv      # 72행 × (조합정보 + E1~E5)
```

### 4.5 중간 분석

- Top-3 조합 선별 (T1 기준 한국어 F1 순위)
- 모듈별 기여도 ablation: 각 Stage별 평균 성능 차이
- C1 vs C2 vs C3 전체 평균 비교

### 4.6 검증 기준

- [ ] 216회 학습 완료 (또는 우선순위 기반 최소 72회)
- [ ] 한국어 EER 10% 미만 조합 1개 이상 확인 (H1 검증)
- [ ] 결과 CSV 완성
- [ ] Top-1 조합 확정 → Phase 5 전달

---

## Phase 5: 실험 B — 증강 효과 검증 (5주차)

### 5.1 실험 설계

**고정**: Phase 4 Top-1 아키텍처 조합 + C2 데이터  
**변수**: 증강 방법만 변경

| ID | 증강 | 적용 대상 | 근거 |
|----|------|-----------|------|
| B1 | 없음 | — | 대조군 (Phase 4 결과 그대로) |
| B2 | Low Frequency Masking | 학습 데이터 | 황규한(2025) EER 6.36% 최고 |
| B3 | Time Masking | 학습 데이터 | 황규한(2025) EER 7.64% |
| B4 | MUSAN Noise Addition | 학습 데이터 | 잡음 강건성 향상 목적 |

### 5.2 증강 구현 상세

```python
# B2: Low Frequency Masking
# 저주파 대역(0~freq_mask_param)을 랜덤 마스킹
torchaudio.transforms.FrequencyMasking(freq_mask_param=20)

# B3: Time Masking
# 시간축 랜덤 구간 마스킹
torchaudio.transforms.TimeMasking(time_mask_param=50)

# B4: MUSAN Noise
# SNR 5~20dB 범위에서 랜덤 잡음 혼합
add_noise(audio, musan_sample, snr=random.uniform(5, 20))
```

### 5.3 제외 증강 (근거)

| 증강 | 제외 사유 |
|------|-----------|
| Echo | 황규한(2025) EER 56.94%, 6배 악화 |
| Volume Control | EER 13.46%, baseline 대비 악화 |

### 5.4 검증 기준

- [ ] B1~B4 각각 학습 완료
- [ ] T1/T2 기준 최적 증강 선정
- [ ] B1 대비 개선폭 정량화
- [ ] 최적 증강 확정 → Phase 6 전달

---

## Phase 6: 실험 C — 강건성 평가 (6주차)

### 6.1 실험 설계

**고정**: Phase 4 Top-1 아키텍처 + Phase 5 최적 증강 + C2 데이터  
**변수**: 평가 조건

| ID | 테스트 | 조건 | 목적 |
|----|--------|------|------|
| P1 | 잡음 강건성 | SNR 5 / 10 / 20 dB Gaussian noise | 실환경 잡음 내성 |
| P2 | 압축 강건성 | MP3 32 / 64 / 128 kbps | 전송·저장 후 탐지 |
| P3 | Cross-dataset | 학습=SpeechFake → 평가=KoAAD (역방향도) | 데이터 일반화 |
| P4 | Cross-generator | unseen TTS 평가 | 미지 생성기 대응력 |

### 6.2 W2V+AASIST 비교 실험

동일 테스트 조건(T1, T2, P1~P4)에서 W2V+AASIST 대형 모델 결과와 직접 비교:
- Accuracy 갭 측정
- Inference Time 비율 측정
- Pareto 프론티어 상 위치 확인

### 6.3 축소 전략 (시간 부족 시)

우선순위: P3 > P1 > P4 > P2
- P3(Cross-dataset)은 핵심 — 반드시 수행
- P2(압축)는 선택적 수행

### 6.4 검증 기준

- [ ] P1~P4 중 최소 P1, P3 완료
- [ ] W2V+AASIST 비교 결과 확보
- [ ] 실전 적용 가능성 판단 근거 확보

---

## Phase 7: 분석 및 시각화 (7주차)

### 7.1 핵심 분석

| 분석 | 내용 | 시각화 |
|------|------|--------|
| Pareto 프론티어 | Accuracy vs Inference Time 2D plot | Scatter + Pareto line |
| 모듈별 Ablation | 각 Stage 교체 시 평균 성능 변화 | Bar chart (4개 Stage) |
| 데이터 조합 효과 | C1 vs C2 vs C3 전체 분포 | Box plot |
| 가설 검증 요약 | H1~H4 수용/기각 판정 | 요약 테이블 |
| 통계 검정 | Paired t-test / Wilcoxon (C1 vs C2) | p-value 보고 |

### 7.2 주요 산출물

1. **전체 결과표** (216행 × 7지표)
2. **Top-5 조합 상세 분석**
3. **Pareto 프론티어 그래프** (Accuracy vs InfTime, Accuracy vs FLOPs)
4. **모듈별 기여도 히트맵**
5. **C1→C2 개선폭 분포**
6. **강건성 결과 레이더 차트**

### 7.3 검증 기준

- [ ] 모든 가설(H1~H4) 수용/기각 판정 완료
- [ ] Pareto 프론티어 그래프 완성
- [ ] 통계적 유의성 검정 완료
- [ ] 그래프/표 논문 투고 수준 완성

---

## Phase 8: 논문 집필 및 발표 (8주차)

### 8.1 논문 구조

```
1. 서론 (연구 배경, 공백, 기여)
2. 관련 연구 (임동훈, SpeechFake, 황규한, KCI 서베이)
3. 방법론 (모듈식 프레임워크, 데이터 조합)
4. 실험 설계 (A/B/C, 평가 체계)
5. 결과 및 분석 (전체 결과, Pareto, ablation)
6. 논의 (가설 검증, 한계, 시사점)
7. 결론 및 향후 연구
```

### 8.2 산출물

- [ ] 논문 초고 (한국어 또는 영어)
- [ ] 발표 슬라이드 (15~20장)
- [ ] 코드 및 실험 로그 정리 (재현 가능)
- [ ] 결과 데이터 공개 준비

---

## 전체 흐름 요약

```
Phase 1 (1주) ─── 데이터 수집·전처리·분할
     │
Phase 2 (2주) ─── 베이스라인 재현 (임동훈 2024)
     │
Phase 3 (3주) ─── 모듈 구현 (72개 조합 파이프라인)
     │
Phase 4 (4주) ─── 실험 A: 216회 그리드 탐색 ← 핵심
     │
Phase 5 (5주) ─── 실험 B: 증강 효과 (Top-1 고정)
     │
Phase 6 (6주) ─── 실험 C: 강건성 + W2V+AASIST 비교
     │
Phase 7 (7주) ─── 분석·시각화·통계
     │
Phase 8 (8주) ─── 논문 집필·발표
```

---

## 위험 관리 체크포인트

| 시점 | 체크포인트 | 미달 시 대응 |
|------|-----------|-------------|
| 1주 말 | 데이터 전체 확보 | SpeechFake 미확보 시 KoAAD+MLAAD로 우선 진행 |
| 2주 말 | 베이스라인 재현 ±2%p | 하이퍼파라미터 재조정, 저자 코드 참고 |
| 4주 말 | EER 10% 미만 1개 이상 | 데이터 조합 효과에 집중, negative result 보고 |
| 6주 말 | 실험 전체 완료 | 실험 C 축소 (P3만 수행) |

---

*이 문서는 연구 진행에 따라 업데이트합니다.*
