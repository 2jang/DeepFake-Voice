"""한국어 딥페이크 음성 탐지 FastAPI 웹앱.

실행:
    python webapp/app.py                 # http://127.0.0.1:8000
    python webapp/app.py --port 8080
    python webapp/app.py --host 0.0.0.0  # LAN 공개 (마이크는 localhost/HTTPS에서만 동작)

기능:
    - 마이크 녹음 / 오디오 파일 업로드 (wav, mp3, flac, ogg, m4a, webm 등)
    - 모델 선택 (experiments/checkpoints의 학습 완료 모델) 및 전체 모델 비교
    - 4초 윈도우 슬라이딩 분석, 구간별 fake 확률, 스펙트로그램 시각화
    - WebSocket 실시간 감시: 2초 윈도우 모델로 0.5초마다 판정 스트리밍

동시성 설계 (다중 사용자):
    - 무거운 작업(디코딩·추론·시각화)은 threadpool로 offload → 이벤트 루프 비블로킹
    - 업로드 분석 구간은 세마포어로 제한 (프로세스당 기본 4) → 과부하 시 초과 요청은 대기
    - 모델 forward는 엔진 내부 락으로 직렬화 (다중 사용자 지연은 미측정)
    - 모델 lazy-load는 이중 검사 락 → 동시 첫 요청에도 1회만 로드
    - 수평 확장이 필요하면 프로세스를 늘리는 대신 별도 추론 서버(큐) 분리 권장
      (uvicorn 멀티워커는 워커마다 모델을 GPU에 중복 적재)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:  # `python webapp/app.py` 실행
    from audio_utils import AudioDecodeError, audio_warnings, load_audio_16k, spectrogram_data_uri, waveform_envelope
    from inference import DEFAULT_MODEL_ID, SAMPLE_RATE, DetectionEngine
except ImportError:  # `uvicorn webapp.app:app` 실행
    from webapp.audio_utils import AudioDecodeError, audio_warnings, load_audio_16k, spectrogram_data_uri, waveform_envelope
    from webapp.inference import DEFAULT_MODEL_ID, SAMPLE_RATE, DetectionEngine

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 60 * 1024 * 1024  # 60MB
MAX_ANALYZE_SEC = 300                # 5분 초과분은 잘라서 분석
MAX_CONCURRENT_ANALYSES = 4          # 동시 분석 상한 (초과분은 대기)

# 실시간 스트리밍
STREAM_DEFAULT_MODEL = "specrnet_2sec"  # 실측 T2 EER 0.34%인 2초 윈도우 모델
STREAM_HOP_DIV = 4                      # hop = window/4 → 2초 윈도우면 0.5초마다 판정
STREAM_MAX_CHUNK_BYTES = 4 * 1024 * 1024

engine = DetectionEngine()
_analysis_sem = threading.BoundedSemaphore(MAX_CONCURRENT_ANALYSES)


def _warmup() -> None:
    """첫 요청 지연 제거: 기본 모델 로드 + 시각화 경로(librosa/matplotlib) 예열."""
    engine.warmup()
    spectrogram_data_uri(np.zeros(16_000, dtype=np.float32))


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_warmup, daemon=True).start()
    yield


app = FastAPI(title="한국어 딥페이크 음성 탐지 데모", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/models")
async def list_models():
    return {
        **engine.device_info(),
        "default_model": DEFAULT_MODEL_ID if DEFAULT_MODEL_ID in engine.specs else None,
        "stream_default_model": STREAM_DEFAULT_MODEL if STREAM_DEFAULT_MODEL in engine.specs else None,
        "models": engine.available_models(),
    }


def _process_request(data: bytes, filename: str, model_ids: list[str]) -> dict:
    """디코딩 → 추론 → 시각화. threadpool에서 실행 (블로킹 허용)."""
    with _analysis_sem:
        wav, orig_sr = load_audio_16k(data)
        if len(wav) < SAMPLE_RATE // 4:
            raise HTTPException(400, "오디오가 너무 짧습니다 (0.25초 이상 필요)")

        truncated = False
        if len(wav) > MAX_ANALYZE_SEC * SAMPLE_RATE:
            wav = wav[: MAX_ANALYZE_SEC * SAMPLE_RATE]
            truncated = True

        results = [engine.predict(wav, mid) for mid in model_ids]

        return {
            "audio": {
                "filename": filename,
                "duration_sec": round(len(wav) / SAMPLE_RATE, 2),
                "original_sr": orig_sr,
                "analyzed_sr": SAMPLE_RATE,
                "truncated": truncated,
                "warnings": audio_warnings(wav),
                "waveform": waveform_envelope(wav),
                "spectrogram": spectrogram_data_uri(wav),
            },
            "compare": len(model_ids) > 1,
            "results": results,
        }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model_id: str = Form(DEFAULT_MODEL_ID),
    compare: bool = Form(False),
):
    data = await file.read()
    if not data:
        raise HTTPException(400, "빈 파일입니다")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"파일이 너무 큽니다 (최대 {MAX_UPLOAD_BYTES // 1024 // 1024}MB)")

    if compare:
        model_ids = engine.compare_ids()
    else:
        if model_id not in engine.specs:
            raise HTTPException(400, f"알 수 없는 모델: {model_id}")
        model_ids = [model_id]

    try:
        # 블로킹 작업은 threadpool로 — 이벤트 루프를 막지 않아 다른 사용자 요청 처리 가능
        return await run_in_threadpool(_process_request, data, file.filename, model_ids)
    except AudioDecodeError as e:
        raise HTTPException(400, f"오디오를 디코딩할 수 없습니다: {e}")


# ── 실시간 스트리밍 ─────────────────────────────────────────────
#
# 프로토콜:
#   C→S  {"type":"start", "sample_rate":16000, "model_id":"specrnet_2sec"}
#   S→C  {"type":"ready", "window_sec":2.0, "hop_sec":0.5, "model_name":...}
#   C→S  binary Float32 PCM chunk (client sample_rate 기준)
#   S→C  {"type":"score", "t":초, "prob_fake":0.97, "inference_ms":8.1}  ← hop마다
#   C→S  {"type":"stop"}
#   S→C  {"type":"summary", "num_scores":N, "mean":..., "max":..., "duration_sec":...}

@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    try:
        cfg = json.loads(await ws.receive_text())
        if cfg.get("type") != "start":
            await ws.close(code=1002)
            return
        model_id = cfg.get("model_id") or STREAM_DEFAULT_MODEL
        if model_id not in engine.specs:
            await ws.send_json({"type": "error", "message": f"알 수 없는 모델: {model_id}"})
            await ws.close(code=1002)
            return
        client_sr = int(cfg.get("sample_rate", SAMPLE_RATE))
        if not 8_000 <= client_sr <= 192_000:
            await ws.send_json({"type": "error", "message": f"지원하지 않는 샘플레이트: {client_sr}"})
            await ws.close(code=1002)
            return

        # 리샘플 비율 (client_sr → 16k). 같으면 통과.
        g = math.gcd(SAMPLE_RATE, client_sr)
        up, down = SAMPLE_RATE // g, client_sr // g

        await run_in_threadpool(engine.warmup, model_id)  # 모델 예열 후 ready
        spec = engine.specs[model_id]
        win = spec["num_samples"]
        hop = win // STREAM_HOP_DIV
        await ws.send_json({
            "type": "ready",
            "model_name": spec["name"],
            "window_sec": win / SAMPLE_RATE,
            "hop_sec": hop / SAMPLE_RATE,
        })

        buf = np.zeros(0, dtype=np.float32)  # 최신 win 샘플만 유지 (링버퍼)
        pending = 0      # 마지막 판정 이후 누적된 새 샘플 수 (16k 기준)
        total = 0        # 총 수신 샘플 수 (16k 기준)
        scores: list[float] = []

        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return

            if msg.get("text"):
                j = json.loads(msg["text"])
                if j.get("type") == "stop":
                    await ws.send_json({
                        "type": "summary",
                        "num_scores": len(scores),
                        "mean": round(float(np.mean(scores)), 4) if scores else None,
                        "max": round(float(np.max(scores)), 4) if scores else None,
                        "duration_sec": round(total / SAMPLE_RATE, 2),
                    })
                    await ws.close()
                    return
                continue

            raw = msg.get("bytes")
            if not raw or len(raw) > STREAM_MAX_CHUNK_BYTES:
                continue
            x = np.frombuffer(raw, dtype=np.float32)
            if up != down and len(x):
                from scipy.signal import resample_poly
                x = resample_poly(x, up, down).astype(np.float32)

            buf = np.concatenate([buf, x])
            if len(buf) > win:
                buf = buf[-win:]
            pending += len(x)
            total += len(x)

            # 새 오디오가 hop만큼 쌓였고 윈도우가 찼으면 최신 윈도우 판정.
            # 청크가 크게 몰려와도 최신 윈도우 1회만 평가 (실시간 semantics).
            if pending >= hop and total >= win:
                pending %= hop
                r = await run_in_threadpool(engine.predict_window, buf.copy(), model_id)
                scores.append(r["prob_fake"])
                await ws.send_json({
                    "type": "score",
                    "t": round(total / SAMPLE_RATE, 2),
                    "prob_fake": r["prob_fake"],
                    "inference_ms": r["inference_ms"],
                })
    except WebSocketDisconnect:
        pass
    except Exception:
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "message": "서버 오류로 스트림이 종료되었습니다"})
            await ws.close(code=1011)


def main() -> None:
    parser = argparse.ArgumentParser(description="딥페이크 음성 탐지 웹앱")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    print(f"\n  딥페이크 음성 탐지 데모: http://{args.host}:{args.port}")
    print(f"  장치: {engine.device_info()}  |  모델 {len(engine.specs)}개\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
