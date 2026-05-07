# Phase 1.1 — 데이터셋 다운로드 스크립트

한국어 딥페이크 음성 탐지 연구에 사용되는 5개 데이터셋의 다운로드 스크립트 모음.

---

## 데이터셋 요약

| # | 데이터셋 | 스크립트 | 역할 | 크기 | 한국어 |
|---|----------|----------|------|------|--------|
| D-a | MLAAD v9 | `download_mlaad.py` | 다국어 딥페이크 음성 (51개 언어, 140개 TTS) | ~109GB | O |
| D-b | M-AILABS | `download_mailabs.py` | 다국어 실제(bonafide) 음성 (9개 언어) | ~127GB | X |
| D-c | SpeechFake-MD | `download_speechfake.py` | 다국어 딥페이크+실제 (46개 언어, 40개 TTS) | 대용량 | O |
| D-d | KoAAD | `download_koaad.py` | 한국어 딥페이크+실제 (6개 TTS) | ~10GB | O |
| D-e | MUSAN | `download_musan.py` | 잡음/음악/환경음 (증강용) | ~11GB | — |

## 사전 준비

```bash
# 공통 의존성
pip install requests tqdm

# MLAAD (HuggingFace)
pip install huggingface_hub
huggingface-cli login

# SpeechFake (ModelScope 또는 HuggingFace)
pip install modelscope    # 방법 1
pip install huggingface_hub  # 방법 2

# KoAAD (Google Drive)
pip install gdown
```

## 사용법

### D-a: MLAAD v9

HuggingFace 호스팅 (`mueller91/MLAAD`). 한국어 필터링 다운로드 지원.

```bash
# 한국어만 다운로드
python scripts/download_mlaad.py -o data/mlaad --languages ko

# 전체 다운로드 (109GB)
python scripts/download_mlaad.py -o data/mlaad

# 사용 가능한 언어 목록
python scripts/download_mlaad.py --list-languages

# 테스트용 tiny 버전
python scripts/download_mlaad.py -o data/mlaad-tiny --tiny
```

| 옵션 | 설명 |
|------|------|
| `--languages ko en ja` | 특정 언어만 다운로드 |
| `--tiny` | MLAAD-tiny (소규모 테스트용) |
| `--selective` | 파일 단위 선택적 다운로드 |
| `--workers 8` | 동시 다운로드 스레드 수 |

### D-b: M-AILABS

caito.de 직접 HTTP 다운로드. 인증 불필요. 이어받기(resume) 지원.

```bash
# 영어(US)만 다운로드 (~13GB)
python scripts/download_mailabs.py -o data/mailabs --languages en_US

# 전체 9개 언어 (~127GB)
python scripts/download_mailabs.py -o data/mailabs

# 사용 가능한 언어 목록
python scripts/download_mailabs.py --list-languages
```

| 옵션 | 설명 |
|------|------|
| `--languages en_US de_DE` | 특정 언어만 다운로드 |
| `--no-extract` | tgz만 다운로드 (압축 해제 안 함) |
| `--keep-archive` | 압축 해제 후 tgz 유지 |

사용 가능한 언어: `de_DE`(237h), `fr_FR`(190h), `it_IT`(127h), `es_ES`(108h), `en_US`(102h), `uk_UK`(87h), `pl_PL`(53h), `ru_RU`(46h), `en_UK`(45h)

### D-c: SpeechFake-MD

ModelScope (`inclusionAI/SPEECHFAKE`) 또는 HuggingFace (`DeepFense/SpeechFake`).

```bash
# ModelScope SDK (권장)
python scripts/download_speechfake.py -o data/speechfake --method modelscope

# HuggingFace
python scripts/download_speechfake.py -o data/speechfake --method huggingface

# Git LFS 클론
python scripts/download_speechfake.py -o data/speechfake --method git

# 한국어만 추출
python scripts/download_speechfake.py -o data/speechfake --extract-languages ko

# 데이터셋 정보
python scripts/download_speechfake.py --info
```

| 옵션 | 설명 |
|------|------|
| `--method modelscope` | ModelScope SDK 다운로드 (기본) |
| `--method huggingface` | HuggingFace 다운로드 |
| `--method git` | Git LFS 클론 (ModelScope) |
| `--extract-languages ko` | 다운로드 후 특정 언어 추출 |

한국어(ko)는 SpeechFake-MD의 9대 주요 언어에 포함. test set에 ko 전용 subset 존재.

### D-d: KoAAD

Google Drive 호스팅. Fake 음성만 포함 — Real 음성(AIHUB)은 별도 수동 다운로드 필요.

```bash
# KoAAD fake 음성 다운로드
python scripts/download_koaad.py -o data/koaad

# GitHub 코드도 함께 클론
python scripts/download_koaad.py -o data/koaad --clone-repo

# 데이터셋 정보
python scripts/download_koaad.py --info
```

| 옵션 | 설명 |
|------|------|
| `--clone-repo` | GitHub 리포지토리 클론 포함 |
| `--method gdown-cli` | gdown CLI 방식 사용 |
| `--structure` | 다운로드된 구조 확인 |

**AIHUB Real 음성 (수동)**:
1. https://www.aihub.or.kr 회원가입
2. "감성 및 발화스타일 동시고려 음성합성 데이터" 검색
3. 데이터 이용 신청 → 승인 대기 (1~3일)
4. 승인 후 다운로드 → `data/aihub/`에 저장

### D-e: MUSAN

OpenSLR 직접 HTTP 다운로드. 인증 불필요. 4개 미러 자동 탐색.

```bash
# noise만 다운로드 (증강 목적이면 이것만으로 충분)
python scripts/download_musan.py -o data/musan --categories noise

# 전체 다운로드 (~11GB)
python scripts/download_musan.py -o data/musan

# EU 미러 사용
python scripts/download_musan.py -o data/musan --mirror eu

# 데이터셋 정보
python scripts/download_musan.py --info
```

| 옵션 | 설명 |
|------|------|
| `--categories noise` | noise만 추출 (증강용) |
| `--categories noise music` | noise + music 추출 |
| `--mirror eu` | EU 미러 사용 |
| `--no-extract` | tar.gz만 다운로드 |

미러: `default`(openslr.org), `eu`(elda.org), `cn`(cn-mirror), `cn2`(magicdatatech)

---

## 데이터 조합 (연구계획서 기준)

다운로드 후 아래 3가지 조합으로 학습 데이터를 구성:

| 조합 | 학습 데이터 | 한국어 | 검증 질문 |
|------|------------|--------|-----------|
| **C1** | MLAAD v9 (ko 제외) + M-AILABS | X | 임동훈(2024) 재현 베이스라인 |
| **C2** | MLAAD v9 (전체) + M-AILABS + SpeechFake-MD ko + KoAAD | O | 한국어 추가 효과 |
| **C3** | SpeechFake-MD ko + KoAAD | O (Only) | 한국어만으로 충분한가? |

## 권장 다운로드 순서

디스크 용량과 연구 우선순위를 고려한 순서:

```
1. KoAAD        (~10GB)   ← 한국어 핵심 데이터, 빠른 다운로드
2. MUSAN noise   (~1GB)   ← 증강용, 소용량
3. MLAAD ko     (일부)    ← 한국어 fake 데이터
4. SpeechFake   (대용량)  ← 한국어 포함 다국어
5. M-AILABS     (~127GB)  ← bonafide, 필요한 언어만 선택 다운로드
```

## 예상 디렉토리 구조

```
data/
├── mlaad/
│   └── fake/
│       ├── ko/
│       ├── en/
│       └── ...
├── mailabs/
│   ├── en_US/
│   ├── de_DE/
│   └── ...
├── speechfake/
│   ├── SpeechFake-BD/
│   └── SpeechFake-MD/
├── koaad/
│   ├── google_cloud_tts/
│   ├── melo_tts/
│   ├── xtts2/
│   └── ...
├── aihub/              ← 수동 다운로드
│   └── ...
└── musan/
    └── musan/
        ├── speech/
        ├── music/
        └── noise/
```

## 라이선스

| 데이터셋 | 라이선스 |
|----------|----------|
| MLAAD v9 | CC-BY-NC 4.0 (비상업) |
| M-AILABS | BSD 3-Clause |
| SpeechFake | CC BY 4.0 |
| KoAAD | 연구 목적 |
| MUSAN | CC BY 4.0 |
