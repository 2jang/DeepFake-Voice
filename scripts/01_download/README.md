# Phase 1.1 — 데이터셋 다운로드

한국어 딥페이크 음성 탐지 연구에 사용되는 5개 데이터셋의 다운로드 스크립트.

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
pip install requests tqdm huggingface_hub gdown
huggingface-cli login
```

## 사용법

### MLAAD v9

```bash
python scripts/01_download/download_mlaad.py -o data/mlaad --languages ko
python scripts/01_download/download_mlaad.py -o data/mlaad              # 전체
python scripts/01_download/download_mlaad.py --list-languages
```

### M-AILABS

```bash
python scripts/01_download/download_mailabs.py -o data/mailabs --languages en_US
python scripts/01_download/download_mailabs.py -o data/mailabs          # 전체
```

사용 가능 언어: `de_DE`(237h), `fr_FR`(190h), `it_IT`(127h), `es_ES`(108h), `en_US`(102h), `uk_UK`(87h), `pl_PL`(53h), `ru_RU`(46h), `en_UK`(45h)

### SpeechFake-MD

```bash
python scripts/01_download/download_speechfake.py -o data/speechfake --method modelscope
python scripts/01_download/download_speechfake.py -o data/speechfake --extract-languages ko
```

### KoAAD

```bash
python scripts/01_download/download_koaad.py -o data/koaad
python scripts/01_download/download_koaad.py -o data/koaad --clone-repo
```

AIHUB Real 음성은 수동 다운로드 필요:
1. https://www.aihub.or.kr 회원가입
2. "감성 및 발화스타일 동시고려 음성합성 데이터" 검색
3. 승인 후 다운로드 → `data/aihub/`

### MUSAN

```bash
python scripts/01_download/download_musan.py -o data/musan --categories noise
python scripts/01_download/download_musan.py -o data/musan              # 전체
```

## 권장 다운로드 순서

```
1. KoAAD        (~10GB)   ← 한국어 핵심, 빠른 다운로드
2. MUSAN noise   (~1GB)   ← 증강용, 소용량
3. MLAAD ko     (일부)    ← 한국어 fake
4. SpeechFake   (대용량)  ← 한국어 포함 다국어
5. M-AILABS     (~127GB)  ← bonafide, 필요한 언어만
```

## 출력 디렉토리 구조

```
data/
├── mlaad/fake/{lang}/{tts_model}/*.wav
├── mailabs/{locale}/by_book/{gender}/{speaker}/{book}/wavs/*.wav
├── speechfake-BD/{tts_system}/{corpus}/*.wav
├── aihub/Training/01.원천데이터/TS_{체}_{번호}/*.wav
└── musan/{music,noise,speech}/{source}/*.wav

D:\speechfake/
├── MD/{tts_system}/{lang}/{voice}/*.wav
└── Real/{corpus}/{split}/{speaker}/*.wav
```

## 라이선스

| 데이터셋 | 라이선스 |
|----------|----------|
| MLAAD v9 | CC-BY-NC 4.0 (비상업) |
| M-AILABS | BSD 3-Clause |
| SpeechFake | CC BY 4.0 |
| KoAAD | 연구 목적 |
| MUSAN | CC BY 4.0 |
