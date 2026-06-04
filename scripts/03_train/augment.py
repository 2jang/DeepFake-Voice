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

import random
import subprocess
import tempfile
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
        self.rng = random.Random(seed)
        print(f"  MUSANNoisePool: {len(self.files)}개 noise 파일 ({category})")

    def sample(self, length: int) -> np.ndarray:
        """랜덤 noise 파일에서 length 샘플만큼 추출.

        noise가 짧으면 반복(tile), 길면 랜덤 크롭.
        """
        path = self.rng.choice(self.files)
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
            start = self.rng.randint(0, len(noise) - length)
            noise = noise[start : start + length]

        return noise


# ---------------------------------------------------------------------------
# Noise functions
# ---------------------------------------------------------------------------

def add_gaussian_noise(wav: np.ndarray, alpha: float = 0.001) -> np.ndarray:
    """Gaussian noise 추가 — 임동훈(2024) Table 3 재현.

    wav_noisy = wav + alpha * N(0, 1)
    """
    noise = np.random.randn(len(wav)).astype(wav.dtype)
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


def apply_codec(
    wav: np.ndarray,
    sr: int = 16_000,
    codec: str = "mp3",
    bitrate: str = "64k",
) -> np.ndarray:
    """ffmpeg를 이용한 코덱 압축/복원 시뮬레이션.

    wav → temp.wav → ffmpeg encode → temp.{codec} → ffmpeg decode → wav
    전화/메신저 통화 환경 시뮬레이션 (mp3 64kbps).
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
            sf.write(f_in.name, wav, sr)
            in_path = f_in.name

        enc_path = in_path.replace(".wav", f".{codec}")
        out_path = in_path.replace(".wav", "_dec.wav")

        # encode
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", in_path,
             "-b:a", bitrate, "-ar", str(sr), "-ac", "1", enc_path],
            check=True, capture_output=True,
        )
        # decode back to wav
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", enc_path,
             "-ar", str(sr), "-ac", "1", out_path],
            check=True, capture_output=True,
        )
        result, _ = sf.read(out_path, dtype="float32")

        # cleanup
        for p in [in_path, enc_path, out_path]:
            Path(p).unlink(missing_ok=True)

        # 길이 맞춤 (codec으로 인한 미세 차이)
        if len(result) > len(wav):
            result = result[: len(wav)]
        elif len(result) < len(wav):
            result = np.pad(result, (0, len(wav) - len(result)))

        return result

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # ffmpeg 없으면 원본 반환
        print(f"  [augment] codec 적용 실패 (ffmpeg 필요): {e}")
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
