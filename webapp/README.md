# 딥페이크 음성 탐지 웹 데모

마이크 녹음 또는 오디오 파일 업로드로 한국어 딥페이크 음성을 실시간 판별하는 FastAPI 웹앱.
`experiments/checkpoints/`의 학습 완료 모델(LCNN·SpecRNet·AASIST·RawNet2)을 사용한다. 데이터와 체크포인트는 저장소에 포함하지 않으며, 추론하려면 별도로 준비해야 한다.

현재는 로컬 데모이며 서비스로 배포하지 않았다. 화면의 EER은 수정 전 데이터 분할로 학습한 체크포인트의 과거 평가값이다. 데이터 분할 문제와 해석 범위는 [검증 기록](../docs/portfolio-verification.md)을 참고한다. 증강 모델의 잡음 환경 성능은 아직 평가하지 않았다.

## 실행

```bash
# 프로젝트 루트에서
pip install fastapi uvicorn python-multipart   # 나머지는 requirements.txt에 포함
python webapp/app.py                            # → http://127.0.0.1:8000
```

옵션: `--port 8080`, `--host 0.0.0.0` (LAN 공개 시 — 단, 마이크 녹음은 브라우저 보안 정책상
localhost 또는 HTTPS에서만 동작)

## 기능

| 기능 | 설명 |
|---|---|
| 🎤 마이크 녹음 | AudioWorklet raw PCM 캡처 → 무손실 WAV 전송 (opus 압축·노이즈제거·AGC 없이 원음 유지) |
| 📁 파일 업로드 | WAV/MP3/FLAC/OGG/M4A/WEBM 등 — soundfile 우선, 실패 시 PyAV(ffmpeg 번들) 디코딩 |
| 📡 실시간 감시 | WebSocket으로 마이크 PCM을 스트리밍 → 2초 윈도우 모델이 새 오디오 0.5초 분량을 받을 때 최신 구간을 판정. EMA 스무딩 판정 + 라이브 차트 |
| 모델 선택 | 기본값 LCNN+증강(MUSAN·Codec). 레지스트리 10종 중 체크포인트가 있는 모델만 제공 |
| 전체 모델 비교 | 사용 가능한 비교 대상 모델을 순서대로 실행해 판정 비교 테이블 표시 |
| 구간별 분석 | 학습 윈도우(기본 4초) 단위 50% overlap 슬라이딩 — 파형 아래 heat strip + 구간별 P(fake) |
| 스펙트로그램 | 모델 입력과 동일한 log power spectrogram (n_fft=512, hop=256) 시각화 |
| 입력 품질 경고 | 무음/클리핑 감지, 5분 초과 시 잘라서 분석 |
| 분석 기록 | 세션 내 판정 이력 (브라우저 메모리, 저장 안 함) |

## 실시간 스트리밍 프로토콜 (`/ws/stream`)

```
C→S  {"type":"start", "sample_rate":16000, "model_id":"specrnet_2sec"}
S→C  {"type":"ready", "window_sec":2.0, "hop_sec":0.5, "model_name":...}
C→S  binary Float32 PCM chunk (클라이언트 sample_rate 기준, ~0.25초 단위)
S→C  {"type":"score", "t":경과초, "prob_fake":0.97, "inference_ms":8.1}   ← 새 오디오 0.5초 분량 수신 기준; 응답 지연 보장 아님
C→S  {"type":"stop"}
S→C  {"type":"summary", "num_scores":N, "mean":..., "max":..., "duration_sec":...}
```

- 클라이언트는 16kHz AudioContext를 우선 시도, 미지원 브라우저는 native rate로 보내면 서버가 `scipy.signal.resample_poly`로 변환
- 오디오 버퍼는 연결마다 최신 모델 입력 길이만 남긴다. 청크가 몰리면 최신 구간을 한 번 판정한다. 점수 이력은 연결 중 계속 쌓이므로 전체 메모리가 일정한 구조는 아니다.
- 표시 판정은 EMA(α=0.35) 스무딩 — 순간 오판 하나에 라벨이 튀지 않도록

## 다중 사용자 / 동시성

- 업로드 분석의 디코딩·추론·시각화와 스트리밍 추론은 스레드 풀에서 실행한다. 스트리밍 리샘플링은 현재 이벤트 루프에서 실행한다.
- 업로드 분석 구간은 세마포어로 프로세스당 최대 4개까지 실행한다. 파일 전체를 읽은 뒤 적용하며, WebSocket 연결 수를 제한하는 장치는 아니다.
- 엔진의 forward 실행은 락으로 직렬화한다. 별도 로딩 락과 캐시로 동시 요청의 모델 중복 로드를 막는다.
- 다중 사용자 부하 시험은 수행하지 않았다. 대기 시간·메모리 상한·처리량을 보장하는 설정으로 해석하지 않는다.
- 여러 프로세스로 실행하면 모델과 락도 프로세스마다 생성된다.

## 판정 규약

- 라벨: `dataset.py`와 동일 — real=1, fake=0. `softmax[:, 0] = P(fake)`
- 판정: 윈도우별 P(fake) 평균 ≥ 0.5 → FAKE. 0.4~0.6은 "판별 불확실" 표시
- 전처리: 학습과 동일 16kHz mono float32, 진폭 정규화 없음, 짧은 입력은 zero-pad

## 구조

```
webapp/
├── app.py           # FastAPI 엔드포인트 (/, /api/models, /api/predict, /ws/stream)
├── inference.py     # 모델 레지스트리 + DetectionEngine (lazy-load, 윈도우 배치 추론)
├── audio_utils.py   # 디코딩(soundfile→PyAV)·리샘플·엔벨로프·스펙트로그램
└── static/
    ├── index.html
    ├── app.js       # UI 로직·시각화
    ├── recorder.js  # AudioWorklet PCM 녹음기 + WAV 인코더
    └── style.css
```
