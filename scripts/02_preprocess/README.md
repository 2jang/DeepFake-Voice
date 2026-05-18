# Phase 1.2–1.4 — 전처리 및 데이터 분할

다운로드된 원본 오디오를 16kHz mono WAV로 변환하고, 품질 검증 후 실험용 데이터 조합(C1/C2/C3)과 평가셋(T1/T2/T3)을 생성합니다.

## 전체 흐름

```
원본 오디오 (data/)
  │
  ├─ [preprocess.py scan]     → metadata/raw_metadata.csv
  │
  ├─ [preprocess.py sample]   → metadata/sampled_metadata.csv (필요량만 추출)
  │
  ├─ [preprocess.py process]  → processed/ (16kHz mono WAV)
  │                           → metadata/processed_metadata.csv
  │
  └─ [build_splits.py]       → manifests/C1/, C2/, C3/, T1, T2, T3
```

## 스크립트

| 파일 | 단계 | 설명 |
|------|------|------|
| `preprocess.py` | 1.2 | 스캔, 샘플링, 리샘플링, 품질검증, 메타데이터 생성 |
| `build_splits.py` | 1.3–1.4 | 데이터 조합 분할, 평가셋 고정 |

## 필요 패키지

```bash
pip install soundfile librosa numpy tqdm pandas scikit-learn
```

## 사용법

### 1단계: 스캔 (메타데이터 생성)

오디오 파일을 처리하지 않고, 디렉토리 구조에서 메타데이터만 추출합니다.

```bash
python scripts/02_preprocess/preprocess.py scan \
    --data-root ./data \
    --meta-dir metadata
```

출력: `metadata/raw_metadata.csv`

| 컬럼 | 설명 |
|------|------|
| file_path | 원본 파일 경로 |
| dataset | 소속 데이터셋 (mlaad, mailabs, speechfake-md, speechfake-real, koaad) |
| label | real / fake |
| language | 언어 코드 (ko, en, de, zh, ...) |
| speaker_id | 화자 식별자 (speaker-disjoint split용) |
| tts_type | TTS/VC 시스템 이름 (real이면 none) |

데이터 소스별 스캐너:

| 스캐너 | 경로 | label |
|--------|------|-------|
| scan_mlaad | `data/mlaad/fake/{lang}/{tts}/` | fake |
| scan_mailabs | `data/mailabs/{locale}/{gender}/{speaker}/` | real |
| scan_speechfake | `data/speechfake/MD/{tts}/{lang}/` | fake |
| scan_speechfake | `data/speechfake/Real/{corpus}/{lang}/` | real |
| scan_koaad | `data/koaad/{google_cloud_tts,xtts2,naver_news}/` | fake |
| scan_aihub | `data/aihub/{Training,Validation}/01.원천데이터/` | real (koaad) |

### 2단계: 샘플링 (필요량만 추출)

조합별 필요량을 계산하여 과도한 real 데이터를 줄입니다.

```bash
python scripts/02_preprocess/preprocess.py sample \
    --meta-dir metadata \
    --seed 42 \
    --buffer 1.3
```

동작 방식:
- **fake 데이터**: 전부 유지 (소수 클래스)
- **mailabs real**: C2 fake 규모 x 버퍼로 캡
- **koaad real (aihub)**: C3 fake 규모 x 버퍼로 캡

출력: `metadata/sampled_metadata.csv`

### 3단계: 리샘플링 + 품질검증

모든 오디오를 16kHz mono WAV로 변환하고 품질 검증을 수행합니다.
`sampled_metadata.csv`가 있으면 우선 사용, 없으면 `raw_metadata.csv` 사용.

```bash
python scripts/02_preprocess/preprocess.py process \
    --workers 8 \
    --output-root processed \
    --meta-dir metadata
```

**품질 검증 기준:**
- 무음 비율 > 95% → 제외
- 클리핑 비율 > 1% → 제외
- 길이 < 0.5초 → 제외
- 파일 손상 → 제외

출력: `metadata/processed_metadata.csv` + `processed/` 폴더

### 4단계: 통계 리포트

```bash
python scripts/02_preprocess/preprocess.py report --meta-dir metadata
```

### 5단계: 데이터 분할

```bash
python scripts/02_preprocess/build_splits.py \
    --meta-dir metadata \
    --output-dir manifests \
    --seed 42
```

## 데이터 조합 정의

| 조합 | 구성 | 목적 |
|------|------|------|
| C1 | MLAAD v9 (ko 제외) + M-AILABS | 다국어 only (베이스라인) |
| C2 | MLAAD v9 (전체) + M-AILABS + SpeechFake ko (MD+Real) | 다국어 + 한국어 |
| C3 | SpeechFake ko (MD+Real) + KoAAD (fake+aihub real) | 한국어 only |

## 평가셋 정의

| 평가셋 | 내용 | 비고 |
|--------|------|------|
| T1 | SpeechFake ko (MD+Real) test split | 한국어 기본 평가 |
| T2 | KoAAD holdout | 한국어 cross-dataset |
| T3 | MLAAD test split (ko 제외) | 다국어 성능 유지 |

## 분할 방식

- **Speaker-disjoint**: 화자가 train/dev/test에 겹치지 않음
- **비율**: train 80% / dev 10% / test 10%
- **클래스 밸런싱**: train만 real:fake = 1:1 (다운샘플링), dev/test는 원본 비율
- **평가셋 우선 분리**: T1/T2/T3 화자를 먼저 분리 후 나머지로 C1/C2/C3 구성

## 출력 구조

```
metadata/
├── raw_metadata.csv           ← scan 결과
├── sampled_metadata.csv       ← sample 결과
└── processed_metadata.csv     ← process 결과

processed/
├── koaad/{speaker_id}/*.wav
├── mailabs/{speaker_id}/*.wav
├── mlaad/{speaker_id}/*.wav
├── speechfake-md/{speaker_id}/*.wav
└── speechfake-real/{speaker_id}/*.wav

manifests/
├── C1/
│   ├── train.csv
│   ├── dev.csv
│   └── test.csv
├── C2/
│   ├── train.csv
│   ├── dev.csv
│   └── test.csv
├── C3/
│   ├── train.csv
│   ├── dev.csv
│   └── test.csv
├── T1_speechfake_ko_test.csv
├── T2_koaad_holdout.csv
└── T3_mlaad_test.csv
```

## 검증 체크리스트

- [ ] 모든 파일 16kHz mono WAV 변환 완료
- [ ] Speaker-disjoint 분할 검증 (train/dev/test 화자 비중복)
- [ ] Class balance 확인 (real:fake ≈ 1:1)
- [ ] C1/C2/C3 매니페스트 CSV 생성 완료
- [ ] T1/T2/T3 평가셋 고정 및 별도 저장
