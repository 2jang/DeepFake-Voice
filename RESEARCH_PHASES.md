# 연구 수행 절차 — Phase별 상세 가이드

**프로젝트**: 한국어 딥페이크 음성 탐지를 위한 경량 모델의 교차 언어 전이 가능성 분석  
**총 기간**: 8주  
**총 실험**: 9회 (3 모델 × 3 데이터 조합)

---

## Phase 1: 데이터 수집 및 전처리 (1~2주차)

### 1.1 데이터셋 다운로드

| 데이터셋 | 출처 | 역할 |
|----------|------|------|
| MLAAD v9 | Fraunhofer ownCloud | 다국어 딥페이크 (51개 언어, 한국어 포함) |
| M-AILABS | caito.de | 다국어 실제 음성 |
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
  → 메타데이터: 파일명, 언어, 라벨(real/fake), 화자ID
```

### 1.3 데이터 조합별 분할

| 조합 | 학습 데이터 구성 | 의미 |
|------|-----------------|------|
| C1 | MLAAD v9 (`lang != 'ko'`) + M-AILABS | 다국어만 (교차 언어 전이 베이스라인) |
| C2 | MLAAD v9 (전체) + M-AILABS + SpeechFake-MD ko | 다국어 + 한국어 |
| C3 | SpeechFake-MD ko + KoAAD | 한국어만 |

분할 방식: Speaker-disjoint train/val/test = 7:1.5:1.5

데이터 규모: 클래스당 ~5,000~10,000개 샘플링

### 1.4 평가셋 고정

| 평가셋 | 내용 | 역할 |
|--------|------|------|
| T1 | SpeechFake-MD ko test split | 한국어 기본 평가 |
| T2 | KoAAD 홀드아웃 | 한국어 cross-dataset |
| T3 | MLAAD test split (ko 제외) | 다국어 성능 유지 확인 |

### 1.5 검증 기준

- [ ] 모든 파일 16kHz mono WAV 변환 완료
- [ ] Speaker-disjoint 분할 검증
- [ ] Class balance 확인 (bonafide:spoof ≈ 1:1)
- [ ] C1/C2/C3 manifest CSV 생성 완료
- [ ] T1/T2/T3 평가셋 고정 및 별도 저장

---

## Phase 2: 모델 환경 구축 (2주차)

### 2.1 모델

| 모델 | 파라미터 | 코드 출처 | 입력 형태 |
|------|---------|-----------|-----------|
| AASIST | 297K | clovaai/aasist | Raw waveform |
| LCNN | 1.5M | 임동훈(2024) 참고 구현 | LFCC / Mel-Spec |
| RawNet2 | 17M | Jungjee/RawNet | Raw waveform |

### 2.2 수행 절차

1. 각 모델 코드 클론, 의존성 설치
2. Dummy 데이터로 forward/backward pass 검증
3. 공통 train/eval 스크립트 작성 (모델만 교체하면 실험 가능)
4. AASIST + C1으로 5 epoch 시범 학습 → 정상 동작 확인

### 2.3 학습 설정

```yaml
optimizer: Adam
lr: 1e-4
batch_size: 32
epochs: 50
early_stopping: patience 10 (val EER)
seed: 42
input: 4초 (64,000 samples)
loss: Weighted BCE
```

### 2.4 폴더 구조

```
experiments/
├── configs/        # 실험별 YAML
├── logs/           # 학습 로그
├── checkpoints/    # best model
└── results.csv     # 9행 결과
```

### 2.5 검증 기준

- [ ] 3개 모델 forward/backward 정상
- [ ] 공통 학습/평가 스크립트 완성
- [ ] 시범 학습 정상 확인

---

## Phase 3: 실험 (3~4주차)

### 3.1 실험 목록

| # | 모델 | 데이터 | 질문 |
|---|------|--------|------|
| 1 | AASIST | C1 | AASIST 교차 언어 전이 |
| 2 | AASIST | C2 | 한국어 추가 효과 |
| 3 | AASIST | C3 | 한국어만으로 충분? |
| 4 | LCNN | C1 | LCNN 교차 언어 전이 (임동훈 비교) |
| 5 | LCNN | C2 | 한국어 추가 효과 |
| 6 | LCNN | C3 | 한국어만으로 충분? |
| 7 | RawNet2 | C1 | RawNet2 교차 언어 전이 |
| 8 | RawNet2 | C2 | 한국어 추가 효과 |
| 9 | RawNet2 | C3 | 한국어만으로 충분? |

### 3.2 실행 순서

```
AASIST (Exp 1→2→3)  ~6시간
LCNN   (Exp 4→5→6)  ~12시간
RawNet2 (Exp 7→8→9)  ~24시간
합계: ~42시간
```

### 3.3 각 실험별 평가

모든 실험은 T1, T2, T3 세 평가셋에서 평가:
- T1 (SpeechFake-MD ko): 한국어 기본 성능
- T2 (KoAAD): 한국어 cross-dataset 일반화
- T3 (MLAAD): 다국어 성능 유지 여부

### 3.4 경량 지표 (모델별 1회)

| 지표 | 측정 |
|------|------|
| Inference Time (ms) | 100회 평균 (warm-up 10회 제외) |
| Model Size (MB) | 파일 크기 |

### 3.5 검증 기준

- [ ] 9회 실험 완료
- [ ] results.csv 작성 (9행 × Acc/EER per T1/T2/T3 + Inference Time + Model Size)
- [ ] 학습 곡선 정상 수렴 확인
- [ ] 체크포인트 저장

---

## Phase 4: 분석 및 시각화 (5~6주차)

### 4.1 분석 항목

| 분석 | 방법 | 연구 질문 |
|------|------|-----------|
| 교차 언어 전이 | C1에서 3모델의 T1/T2(한국어) 성능 | RQ1 |
| 한국어 추가 효과 | C1→C2 EER 개선폭 | RQ2 |
| 모델별 전이 능력 | C1 성능 + 개선폭 비교 | RQ3 |
| 다국어 vs 한국어 | C2 vs C3 | 보조 |
| 다국어 성능 유지 | C1→C2에서 T3 변화 | 보조 |
| 임동훈(2024) 비교 | Exp-4 vs 임동훈 LCNN 결과 | 선행 연구 |

### 4.2 가설 검증

| 가설 | 기준 |
|------|------|
| H1: C1 한국어 EER 30%+ (전이 실패) | T1 EER > 30% |
| H2: C2 한국어 EER 10% 미만 | T1 EER < 10% |
| H3: AASIST 교차 전이 가장 우수 | C1에서 AASIST EER 최저 |
| H4: C2 > C3 | C2 EER < C3 EER |

### 4.3 시각화

1. 결과표 (9행)
2. 교차 언어 전이 바 차트 (C1에서 3모델 EER)
3. 한국어 추가 효과 그래프 (C1→C2 변화)
4. 데이터 조합 비교 (C1/C2/C3 grouped bar)
5. 가설 검증 요약표

### 4.4 검증 기준

- [ ] 분석 6개 항목 완료
- [ ] 시각화 5개 완성
- [ ] H1~H4 판정 완료

---

## Phase 5: 논문 및 발표 (7~8주차)

### 5.1 논문 구조

```
1. 서론 — 위협, 한국어 공백, 교차 언어 전이 필요성
2. 관련 연구 — 임동훈(2024), SpeechFake(2025)
3. 방법 — 모델 3종, 데이터 C1/C2/C3, 실험 설계
4. 결과 — 결과표, 전이 분석, 한국어 추가 효과
5. 논의 — 가설 검증, 한계
6. 결론
```

### 5.2 산출물

- [ ] 논문 초고
- [ ] 발표 슬라이드 (15~20장)
- [ ] 코드 정리

---

## 전체 흐름

```
Phase 1 (1~2주) ── 데이터 4개 준비, C1/C2/C3 분할, T1/T2/T3 확정
     │
Phase 2 (2주)   ── 모델 3개 세팅, 시범 학습
     │
Phase 3 (3~4주) ── 9회 실험 ← 핵심
     │
Phase 4 (5~6주) ── 교차 언어 전이 분석, 시각화
     │
Phase 5 (7~8주) ── 논문·발표
```

---

## 위험 관리

| 시점 | 체크 | 미달 시 |
|------|------|---------|
| 2주 말 | 데이터 + 모델 환경 준비 완료 | SpeechFake 미확보 시 KoAAD+MLAAD로 진행 |
| 4주 말 | 9회 실험 완료 | RawNet2 제외, AASIST+LCNN 6회로 진행 |
| 6주 말 | 분석 완료 | 핵심 분석(RQ1~3)만 완성 |

---

*이 문서는 연구 진행에 따라 업데이트합니다.*
