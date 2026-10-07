"""오디오 디코딩·리샘플·시각화 유틸.

- 디코딩: soundfile(wav/flac/ogg/mp3) 우선, 실패 시 PyAV(webm/opus/m4a/aac 등).
  PyAV는 ffmpeg를 번들하므로 시스템 ffmpeg 설치가 불필요하다 (requirements의 av).
- 전처리 규약은 scripts/02_preprocess/preprocess.py와 동일:
  16kHz mono float32, 진폭 정규화 없음.
"""

from __future__ import annotations

import base64
import io
import math

import numpy as np
import soundfile as sf

TARGET_SR = 16_000

# 스펙트로그램은 모델 입력과 동일 파라미터 (specrnet.py / lcnn.py)
N_FFT = 512
HOP_LENGTH = 256
SPEC_MAX_COLS = 2400  # 긴 오디오는 시간축 max-pool로 축소


class AudioDecodeError(Exception):
    """지원하지 않는 형식이거나 손상된 오디오."""


def _to_mono(wav: np.ndarray) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return np.ascontiguousarray(wav, dtype=np.float32)


def _decode_soundfile(data: bytes) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return _to_mono(wav), sr


def _decode_av(data: bytes) -> tuple[np.ndarray, int]:
    """PyAV로 디코딩 + 16kHz mono float32 리샘플을 한 번에 수행."""
    import av

    chunks: list[np.ndarray] = []
    orig_sr = 0
    try:
        with av.open(io.BytesIO(data)) as container:
            if not container.streams.audio:
                raise AudioDecodeError("오디오 스트림이 없습니다")
            orig_sr = container.streams.audio[0].rate or 0
            resampler = av.AudioResampler(format="flt", layout="mono", rate=TARGET_SR)
            for frame in container.decode(audio=0):
                for rf in resampler.resample(frame):
                    chunks.append(rf.to_ndarray().reshape(-1))
            for rf in resampler.resample(None):  # flush
                chunks.append(rf.to_ndarray().reshape(-1))
    except AudioDecodeError:
        raise
    except Exception as e:
        raise AudioDecodeError("지원하지 않는 형식이거나 손상된 파일입니다") from e
    if not chunks:
        raise AudioDecodeError("디코딩된 오디오 프레임이 없습니다")
    return np.concatenate(chunks).astype(np.float32), orig_sr or TARGET_SR


def load_audio_16k(data: bytes) -> tuple[np.ndarray, int]:
    """바이트 → (16kHz mono float32 waveform, 원본 샘플레이트)."""
    try:
        wav, sr = _decode_soundfile(data)
    except Exception:
        wav, orig_sr = _decode_av(data)  # av 경로는 이미 16kHz
        return wav, orig_sr
    if sr != TARGET_SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    return wav.astype(np.float32), sr


def audio_warnings(wav: np.ndarray) -> list[str]:
    """입력 품질 경고 코드 목록 (프론트에서 한국어 메시지로 변환)."""
    warns = []
    if len(wav) == 0:
        return ["empty"]
    rms = float(np.sqrt(np.mean(wav ** 2)))
    if rms < 1e-4:
        warns.append("near_silent")
    if float(np.mean(np.abs(wav) >= 0.985)) > 0.01:
        warns.append("clipping")
    return warns


def waveform_envelope(wav: np.ndarray, bins: int = 1200) -> dict:
    """캔버스 렌더용 min/max 엔벨로프 다운샘플."""
    n = len(wav)
    if n == 0:
        return {"min": [], "max": []}
    bins = min(bins, n)
    edges = np.linspace(0, n, bins + 1, dtype=np.int64)
    mins, maxs = [], []
    for i in range(bins):
        seg = wav[edges[i]: edges[i + 1]]
        if len(seg) == 0:
            mins.append(0.0)
            maxs.append(0.0)
        else:
            mins.append(round(float(seg.min()), 4))
            maxs.append(round(float(seg.max()), 4))
    return {"min": mins, "max": maxs}


_CMAP = None


def _get_cmap():
    """magma 컬러맵 lazy 로드 (pyplot 없이 — 폰트 캐시 초기화 비용 회피)."""
    global _CMAP
    if _CMAP is None:
        from matplotlib import colormaps
        _CMAP = colormaps["magma"]
    return _CMAP


def spectrogram_data_uri(wav: np.ndarray) -> str:
    """모델 입력과 동일한 log power spectrogram을 JPEG data URI로 렌더.

    matplotlib figure 대신 컬러맵→PIL 직접 렌더링 (수 초 → 수십 ms).
    축 라벨은 프론트에서 표시한다 (x: 0~duration, y: 0~8kHz).
    """
    import librosa
    from PIL import Image

    if len(wav) < N_FFT:
        wav = np.pad(wav, (0, N_FFT - len(wav)))
    S = librosa.stft(wav, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann", center=False)
    logp = np.log(np.abs(S) ** 2 + 1e-8)  # (257, T)

    # 긴 오디오: 시간축 max-pool로 이미지 폭 제한
    T = logp.shape[1]
    if T > SPEC_MAX_COLS:
        k = math.ceil(T / SPEC_MAX_COLS)
        pad = (-T) % k
        if pad:
            logp = np.pad(logp, ((0, 0), (0, pad)), constant_values=logp.min())
        logp = logp.reshape(logp.shape[0], -1, k).max(axis=2)

    lo, hi = np.percentile(logp, [2.0, 99.5])
    norm = np.clip((logp - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgb = (_get_cmap()(norm)[..., :3] * 255).astype(np.uint8)
    img = Image.fromarray(np.flipud(rgb))  # 저주파가 아래로
    if img.height < 300:
        img = img.resize((img.width, 300), Image.BILINEAR)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
