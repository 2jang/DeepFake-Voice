"""딥페이크 음성 탐지 추론 엔진.

scripts/03_train의 모델 정의와 experiments/checkpoints의 학습 완료 가중치를
로드해 임의 길이 오디오의 fake/real 확률을 계산한다.

- 입력: 16kHz mono float32 waveform (길이 제한 없음)
- 윈도우 분석: 모델 학습 길이(num_samples) 단위, 50% overlap 슬라이딩
- 라벨 규약: dataset.py LABEL_MAP {"real": 1, "fake": 0} → softmax[:, 0] = P(fake)
- 체크포인트: torch.compile 학습분의 '_orig_mod.' 접두사 제거 후 로드 (train.py와 동일)
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "03_train"))

from models import get_model  # noqa: E402

SAMPLE_RATE = 16_000
BATCH_SIZE = 64  # 윈도우 배치 추론 상한

# 웹앱에 노출할 모델 목록. t2_eer는 results_v2.csv의 T2(KoAAD holdout) EER(%).
# group은 프론트 <optgroup> 구분용, compare=True인 모델만 "전체 모델 비교"에 포함.
MODEL_SPECS: list[dict] = [
    dict(
        id="lcnn_aug_both", arch="lcnn", ckpt="lcnn_call_aug_both_best.pt",
        num_samples=64_000, name="LCNN + 증강 (MUSAN·Codec)", group="추천",
        params="1.37M", t2_eer=0.34, compare=True,
        desc="노이즈·코덱 증강 학습 — 마이크 등 실환경 입력에 가장 강건 (권장 기본값)",
    ),
    dict(
        id="lcnn", arch="lcnn", ckpt="lcnn_call_best.pt",
        num_samples=64_000, name="LCNN", group="기본 모델 (C_all · 4초)",
        params="1.37M", t2_eer=0.19, compare=True,
        desc="Max Feature Map 기반 경량 CNN — 연구 종합 최고 성능",
    ),
    dict(
        id="specrnet", arch="specrnet", ckpt="specrnet_call_best.pt",
        num_samples=64_000, name="SpecRNet", group="기본 모델 (C_all · 4초)",
        params="416K", t2_eer=0.16, compare=True,
        desc="스펙트로그램 ResNet + FMS Attention — 초경량 고성능",
    ),
    dict(
        id="specrnet_aug_both", arch="specrnet", ckpt="specrnet_call_aug_both_best.pt",
        num_samples=64_000, name="SpecRNet + 증강 (MUSAN·Codec)", group="기본 모델 (C_all · 4초)",
        params="416K", t2_eer=0.45, compare=True,
        desc="SpecRNet의 노이즈·코덱 증강 학습 버전",
    ),
    dict(
        id="aasist", arch="aasist", ckpt="aasist_call_best.pt",
        num_samples=64_000, name="AASIST", group="기본 모델 (C_all · 4초)",
        params="229K", t2_eer=1.50, compare=True,
        desc="최소 파라미터(229K) — SincConv + Graph Attention",
    ),
    dict(
        id="rawnet2", arch="rawnet2", ckpt="rawnet2_call_best.pt",
        num_samples=64_000, name="RawNet2", group="기본 모델 (C_all · 4초)",
        params="18.5M", t2_eer=0.37, compare=True,
        desc="SincConv + GRU 시계열 처리 — 대형 베이스라인",
    ),
    dict(
        id="lcnn_2sec", arch="lcnn", ckpt="lcnn_call_2sec_best.pt",
        num_samples=32_000, name="LCNN (2초 윈도우)", group="윈도우 길이 실험",
        params="1.37M", t2_eer=None, compare=False,
        desc="저지연 실험 — 2초 단위 분석",
    ),
    dict(
        id="lcnn_8sec", arch="lcnn", ckpt="lcnn_call_8sec_best.pt",
        num_samples=128_000, name="LCNN (8초 윈도우)", group="윈도우 길이 실험",
        params="1.37M", t2_eer=None, compare=False,
        desc="장문 컨텍스트 실험 — 8초 단위 분석",
    ),
    dict(
        id="specrnet_2sec", arch="specrnet", ckpt="specrnet_call_2sec_best.pt",
        num_samples=32_000, name="SpecRNet (2초 윈도우)", group="윈도우 길이 실험",
        params="416K", t2_eer=0.34, compare=False,
        desc="저지연 실험 — 2초 단위 분석",
    ),
    dict(
        id="specrnet_8sec", arch="specrnet", ckpt="specrnet_call_8sec_best.pt",
        num_samples=128_000, name="SpecRNet (8초 윈도우)", group="윈도우 길이 실험",
        params="416K", t2_eer=0.13, compare=False,
        desc="장문 컨텍스트 실험 — 8초 단위 분석",
    ),
]

DEFAULT_MODEL_ID = "lcnn_aug_both"


class DetectionEngine:
    """체크포인트 lazy-load + 캐시. 스레드 안전(추론 직렬화)."""

    def __init__(self, ckpt_dir: Path | str | None = None, device: str | None = None):
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else PROJECT_ROOT / "experiments" / "checkpoints"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._cache: dict[str, torch.nn.Module] = {}
        self._lock = threading.Lock()        # GPU 추론 직렬화
        self._load_lock = threading.Lock()   # 모델 lazy-load 직렬화 (동시 요청 시 중복 로드 방지)
        # 체크포인트 파일이 실제 존재하는 모델만 노출
        self.specs: dict[str, dict] = {
            s["id"]: s for s in MODEL_SPECS if (self.ckpt_dir / s["ckpt"]).exists()
        }

    # ── 정보 조회 ──────────────────────────────────────────────

    def device_info(self) -> dict:
        info = {"device": self.device.type}
        if self.device.type == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name(self.device)
        return info

    def available_models(self) -> list[dict]:
        out = []
        for s in self.specs.values():
            out.append({
                "id": s["id"], "name": s["name"], "group": s["group"],
                "desc": s["desc"], "params": s["params"], "t2_eer": s["t2_eer"],
                "window_sec": s["num_samples"] / SAMPLE_RATE,
                "loaded": s["id"] in self._cache,
            })
        return out

    def compare_ids(self) -> list[str]:
        return [s["id"] for s in self.specs.values() if s["compare"]]

    # ── 모델 로드 ──────────────────────────────────────────────

    def _load(self, model_id: str) -> torch.nn.Module:
        if model_id in self._cache:
            return self._cache[model_id]
        with self._load_lock:
            if model_id in self._cache:  # double-checked: 대기 중 다른 스레드가 로드했을 수 있음
                return self._cache[model_id]
            spec = self.specs[model_id]
            model = get_model(spec["arch"], num_classes=2, num_samples=spec["num_samples"])
            state = torch.load(self.ckpt_dir / spec["ckpt"], map_location="cpu", weights_only=True)
            consume_prefix_in_state_dict_if_present(state, "_orig_mod.")
            model.load_state_dict(state)
            model.eval().to(self.device)
            self._cache[model_id] = model
            return model

    def warmup(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        """기본 모델을 미리 로드하고 더미 추론 1회 (첫 요청 지연 제거)."""
        if model_id not in self.specs:
            return
        spec = self.specs[model_id]
        model = self._load(model_id)
        dummy = torch.zeros(1, spec["num_samples"], device=self.device)
        with torch.inference_mode():
            model(dummy)

    # ── 추론 ──────────────────────────────────────────────────

    def window_samples(self, model_id: str) -> int:
        return self.specs[model_id]["num_samples"]

    def predict_window(self, wav: np.ndarray, model_id: str) -> dict:
        """스트리밍용 경량 추론: 정확히 윈도우 길이 하나만 처리.

        wav 길이가 num_samples와 다르면 pad/뒤쪽 crop.
        """
        spec = self.specs[model_id]
        model = self._load(model_id)
        n = spec["num_samples"]
        if len(wav) < n:
            wav = np.pad(wav.astype(np.float32), (0, n - len(wav)))
        elif len(wav) > n:
            wav = wav[-n:]
        x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))[None, :]

        with self._lock, torch.inference_mode():
            t0 = time.perf_counter()
            logits = model(x.to(self.device))
            prob_fake = float(torch.softmax(logits, dim=1)[0, 0])
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            infer_ms = (time.perf_counter() - t0) * 1000.0

        return {"prob_fake": round(prob_fake, 4), "inference_ms": round(infer_ms, 1)}

    @staticmethod
    def _make_windows(wav: np.ndarray, num_samples: int) -> tuple[np.ndarray, list[int]]:
        """50% overlap 윈도우 분할. 짧으면 zero-pad(학습과 동일), 마지막은 끝에 정렬."""
        if len(wav) <= num_samples:
            padded = np.pad(wav, (0, num_samples - len(wav)))
            return padded[None, :], [0]
        hop = num_samples // 2
        starts = list(range(0, len(wav) - num_samples + 1, hop))
        if starts[-1] + num_samples < len(wav):
            starts.append(len(wav) - num_samples)
        windows = np.stack([wav[s: s + num_samples] for s in starts])
        return windows, starts

    def predict(self, wav: np.ndarray, model_id: str) -> dict:
        """wav: 16kHz mono float32. 반환: 전체/윈도우별 P(fake)와 메타 정보."""
        if model_id not in self.specs:
            raise KeyError(f"unknown model_id: {model_id}")
        spec = self.specs[model_id]
        model = self._load(model_id)
        num_samples = spec["num_samples"]
        duration = len(wav) / SAMPLE_RATE

        windows, starts = self._make_windows(wav.astype(np.float32), num_samples)
        batch = torch.from_numpy(windows)

        with self._lock, torch.inference_mode():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
            probs_list = []
            for i in range(0, len(batch), BATCH_SIZE):
                chunk = batch[i: i + BATCH_SIZE].to(self.device)
                logits = model(chunk)
                probs_list.append(torch.softmax(logits, dim=1).cpu())
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            infer_ms = (time.perf_counter() - t0) * 1000.0

        probs = torch.cat(probs_list).numpy()  # (W, 2), [:, 0]=fake, [:, 1]=real
        p_fake_windows = probs[:, 0]
        p_fake = float(p_fake_windows.mean())

        window_out = []
        for s, p in zip(starts, p_fake_windows):
            window_out.append({
                "start": round(s / SAMPLE_RATE, 2),
                "end": round(min((s + num_samples) / SAMPLE_RATE, duration), 2),
                "prob_fake": round(float(p), 4),
            })
        worst = max(window_out, key=lambda w: w["prob_fake"])

        return {
            "model_id": model_id,
            "model_name": spec["name"],
            "params": spec["params"],
            "t2_eer": spec["t2_eer"],
            "window_sec": num_samples / SAMPLE_RATE,
            "verdict": "fake" if p_fake >= 0.5 else "real",
            "prob_fake": round(p_fake, 4),
            "prob_real": round(1.0 - p_fake, 4),
            "windows": window_out,
            "worst_window": worst,
            "num_windows": len(window_out),
            "inference_ms": round(infer_ms, 1),
            "device": self.device.type,
        }
