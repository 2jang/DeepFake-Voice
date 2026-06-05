"""오디오 증강 유틸리티.

기능:
  1. Gaussian noise 추가 — 임동훈(2024) Table 3 재현
  2. MUSAN noise mixing — SNR 기반 환경 잡음
  3. Codec 압축 시뮬레이션 — 전화/메신저 통화 환경
  4. 조합 증강

사용법 (독립 테스트):
    python scripts/03_train/augment.py --test data/musan/musan
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# MUSAN Noise Pool
# ---------------------------------------------------------------------------

class MUSANNoisePool:
    """MUSAN noise 파일을 관리하는 풀.

    초기화 시 파일 경로만 수집하고, sample() 호출 시 lazy 로드.
    """

    def __init__(
        self,
        musan_dir: str | Path,
        category: str = "noise",
        sr: int = 16_000,
        seed: int = 42,
    ):
        musan_path = Path(musan_dir)
        # musan/{category}/**/*.wav 구조
        self.files = sorted(musan_path.glob(f"{category}/**/*.wav"))
        if not self.files:
            # category가 이미 경로에 포함된 경우
            self.files = sorted(musan_path.glob("**/*.wav"))
        if not self.files:
            raise FileNotFoundError(
                f"MUSAN noise 파일을 찾을 수 없습니다: {musan_path}/{category}/**/*.wav"
            )
        self.sr = sr
        self._rng = np.random.default_rng(seed)
        print(f"  MUSANNoisePool: {len(self.files)}개 noise 파일 ({category})")

    def sample(self, length: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """랜덤 noise 파일에서 length 샘플만큼 추출.

        noise가 짧으면 반복(tile), 길면 랜덤 크롭.

        rng를 넘기면 파일 선택·크롭 위치가 그 generator로 결정된다. 테스트 시
        샘플 인덱스 기반 generator를 넘기면 워커/모델/실행 순서와 무관하게
        동일한 노이즈가 재현된다(모델 간 동일 입력 보장). None이면 내부 generator.
        """
        gen = rng if rng is not None else self._rng
        path = self.files[int(gen.integers(len(self.files)))]
        noise, sr = sf.read(str(path), dtype="float32")
        if noise.ndim > 1:
            noise = noise[:, 0]

        # SR 불일치 시 간단 리샘플 (정수 비율만 지원, 아니면 그냥 사용)
        if sr != self.sr:
            ratio = self.sr / sr
            indices = np.arange(0, len(noise), 1.0 / ratio).astype(int)
            indices = indices[indices < len(noise)]
            noise = noise[indices]

        # 길이 맞추기
        if len(noise) < length:
            repeats = (length // len(noise)) + 1
            noise = np.tile(noise, repeats)
        if len(noise) > length:
            start = int(gen.integers(0, len(noise) - length + 1))
            noise = noise[start : start + length]

        return noise


# ---------------------------------------------------------------------------
# Noise functions
# ---------------------------------------------------------------------------

def add_gaussian_noise(
    wav: np.ndarray, alpha: float = 0.001, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Gaussian noise 추가 — 임동훈(2024) Table 3 재현.

    wav_noisy = wav + alpha * N(0, 1)

    rng를 넘기면 그 generator로 노이즈를 생성하여 재현성을 보장한다.
    None이면 전역 np.random(비결정적)을 사용한다.
    """
    gen = rng if rng is not None else np.random
    noise = gen.standard_normal(len(wav)).astype(wav.dtype)
    return wav + alpha * noise


def add_noise_snr(wav: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """SNR 기반 노이즈 혼합.

    Args:
        wav: 원본 음성 (float32)
        noise: 노이즈 신호 (같은 길이)
        snr_db: 목표 SNR (dB). 높을수록 깨끗함.

    공식: mixed = wav + noise * (rms_wav / rms_noise) * 10^(-snr_db/20)
    """
    rms_wav = np.sqrt(np.mean(wav ** 2) + 1e-10)
    rms_noise = np.sqrt(np.mean(noise ** 2) + 1e-10)
    scale = (rms_wav / rms_noise) * (10.0 ** (-snr_db / 20.0))
    mixed = wav + noise * scale
    # 클리핑 방지
    peak = np.abs(mixed).max()
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(wav.dtype)


def _parse_bitrate(bitrate: str | int) -> int:
    """'64k' → 64000, '128k' → 128000, 192000 → 192000."""
    if isinstance(bitrate, (int, float)):
        return int(bitrate)
    s = str(bitrate).strip().lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1_000)
    if s.endswith("m"):
        return int(float(s[:-1]) * 1_000_000)
    return int(s)


def apply_codec(
    wav: np.ndarray,
    sr: int = 16_000,
    codec: str = "mp3",
    bitrate: str = "64k",
) -> np.ndarray:
    """In-memory 코덱 압축/복원 시뮬레이션 (PyAV).

    wav를 메모리에서 mp3로 인코딩한 뒤 다시 디코딩하여 코덱 아티팩트를
    재현한다. subprocess/임시파일을 쓰지 않으므로:
      - 빠르다 (학습 중 on-the-fly 증강에 적합 — 프로세스 spawn/디스크 I/O 없음)
      - Windows에서도 안전하다 (NamedTemporaryFile 잠금 문제 없음)
      - PyAV가 ffmpeg를 번들하므로 시스템 ffmpeg 버전에 의존하지 않는다

    전화/메신저 통화 환경 시뮬레이션 (mp3 64kbps 기본).
    디코더가 인코더 delay를 보정하므로 시간 정렬은 유지되며(best-lag≈0),
    프레임 경계로 인한 미세한 길이 차이만 입력 길이에 맞춰 truncate/pad 한다.
    """
    import av  # lazy: codec 증강을 쓸 때만 필요 (gaussian/musan 경로엔 불필요)

    encoder = {"mp3": "libmp3lame"}.get(codec, codec)
    try:
        wav = np.ascontiguousarray(wav, dtype=np.float32)
        bit_rate = _parse_bitrate(bitrate)

        # --- encode → mp3 (in-memory) ---
        buf = io.BytesIO()
        with av.open(buf, mode="w", format=codec) as oc:
            # layout="mono" 필수: 미지정 시 stereo로 업믹스되어 joint-stereo
            # 인코딩 후 단일 채널만 뽑으면 신호가 깨진다(전화/메신저도 mono가 맞음).
            ostream = oc.add_stream(encoder, rate=sr, layout="mono")
            ostream.bit_rate = bit_rate
            if encoder == "libmp3lame":
                # 증강 목적이라 인코더 속도 우선 (아티팩트는 bitrate가 좌우).
                try:
                    ostream.codec_context.options = {"compression_level": "9"}
                except Exception:
                    pass
            frame = av.AudioFrame.from_ndarray(
                wav.reshape(1, -1), format="fltp", layout="mono"
            )
            frame.sample_rate = sr
            for pkt in ostream.encode(frame):
                oc.mux(pkt)
            for pkt in ostream.encode():  # flush encoder
                oc.mux(pkt)

        # --- decode → waveform (디코더 native fmt를 직접 변환, resampler 불필요) ---
        buf.seek(0)
        chunks: list[np.ndarray] = []
        with av.open(buf, mode="r") as ic:
            for frame in ic.decode(audio=0):
                a = frame.to_ndarray()
                a = a[0] if a.ndim > 1 else a          # mono: 첫 plane
                if a.dtype.kind == "i":                # s16p 등 정수 포맷 → [-1,1]
                    a = a.astype(np.float32) / np.iinfo(a.dtype).max
                else:
                    a = a.astype(np.float32)
                chunks.append(a)

        if not chunks:
            return wav
        out = np.concatenate(chunks)

        # 길이 맞춤 (프레임 경계로 인한 미세한 길이 차이 보정)
        if len(out) >= len(wav):
            out = out[: len(wav)]
        else:
            out = np.pad(out, (0, len(wav) - len(out)))
        return out

    except Exception as e:  # 코덱/PyAV 문제 시 원본 반환 (학습 중단 방지)
        print(f"  [augment] codec 적용 실패: {e}")
        return wav


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="증강 유틸리티 테스트")
    parser.add_argument("--test", type=str, help="MUSAN 디렉토리 경로")
    args = parser.parse_args()

    # 기본 테스트: 합성 신호
    sr = 16_000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    wav = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz 사인파

    print("=== Gaussian noise ===")
    noisy = add_gaussian_noise(wav, alpha=0.001)
    snr = 10 * np.log10(np.mean(wav ** 2) / (np.mean((noisy - wav) ** 2) + 1e-10))
    print(f"  alpha=0.001, 실효 SNR: {snr:.1f} dB")

    print("\n=== Codec (mp3 64k) ===")
    coded = apply_codec(wav, sr=sr, codec="mp3", bitrate="64k")
    codec_snr = 10 * np.log10(np.mean(wav ** 2) / (np.mean((coded - wav) ** 2) + 1e-10))
    print(f"  MP3 64kbps SNR: {codec_snr:.1f} dB")

    if args.test:
        print(f"\n=== MUSAN noise ({args.test}) ===")
        pool = MUSANNoisePool(args.test, category="noise")
        noise = pool.sample(len(wav))
        for snr_db in [20, 10, 5]:
            mixed = add_noise_snr(wav, noise, snr_db)
            actual = 10 * np.log10(np.mean(wav ** 2) / (np.mean((mixed - wav) ** 2) + 1e-10))
            print(f"  목표 SNR={snr_db}dB → 실효 SNR: {actual:.1f} dB")

    print("\nDone.")
