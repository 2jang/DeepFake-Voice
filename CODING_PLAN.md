# Coding Plan — Phase 4~6 구현 계획

**기반 코드**: `scripts/03_train/` (train.py, dataset.py, metrics.py, build_results.py, models/)
**목표**: Phase 4(지표 보강) → Phase 5(강건성 실험) → Phase 6(시각화) 코드 구현

---

## 현재 코드 구조

```
scripts/03_train/
├── train.py              # 학습 + 평가 (체크포인트→T1/T2/T3 평가 포함)
├── build_results.py      # 체크포인트 재평가 → results.csv 생성
├── dataset.py            # AudioDataset (manifest CSV → 파형 로드, pad/crop)
├── metrics.py            # compute_eer, compute_all (EER, Acc, F1, Prec, Rec, AUC)
├── analyze.py            # results.csv 읽어서 SOTA 비교 출력
├── sota.py               # ASVspoof 2019 SOTA 레퍼런스
├── models/               # AASIST, LCNN, SpecRNet, RawNet2
├── configs/              # 13개 실험 YAML (aasist_c1~rawnet2_call)
├── merge_manifests.py    # C_all manifest 병합
├── launch_tb.py          # TensorBoard 런처
└── verify_models.py      # 모델 forward 검증

scripts/01_download/
└── download_musan.py     # MUSAN 다운로드 (이미 작성됨)

scripts/02_preprocess/
├── preprocess.py         # 오디오 전처리
└── build_splits.py       # train/dev/test 분할
```

**핵심 흐름**:
- `train.py`: YAML config 읽기 → AudioDataset 생성 → 학습 → best checkpoint 저장 → T1/T2/T3 evaluate → results.csv 추가
- `build_results.py`: 학습 없이 기존 checkpoint로 evaluate → results.csv 생성
- `evaluate()` 함수 (train.py L75-93): model → scores → `compute_all(labels, scores)` → dict 반환

**중요 인터페이스**:
- `AudioDataset(manifest_path, num_samples, is_train, seed)` → (wav_tensor, label)
- `compute_all(labels, scores)` → `{eer, accuracy, f1, precision, recall, auc}`
- YAML config: `experiment`, `model.name`, `data.{train/dev/eval_manifests}`, `data.num_samples`, `training.*`

---

## Phase 4: 평가 지표 보강

### 4-1. metrics.py 확장

**파일**: `scripts/03_train/metrics.py`
**변경**: `compute_mindcf`, `compute_fpr_fnr`, `compute_eer_threshold`, `compute_fpr_fnr_at_eer` 함수 추가, `compute_all` 확장 (6→11개 지표)

```python
# 추가할 함수 1: minDCF
def compute_mindcf(
    labels: np.ndarray,
    scores: np.ndarray,
    c_miss: float = 2.0,   # 임동훈(2024) 설정
    c_fa: float = 1.0,
    p_target: float = 0.5,
) -> float:
    """ASVspoof 표준 minDCF.
    
    DCF(θ) = c_miss * p_target * FNR(θ) + c_fa * (1-p_target) * FPR(θ)
    minDCF = min over all θ of DCF(θ)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    dcf = c_miss * p_target * fnr + c_fa * (1 - p_target) * fpr
    # 정규화
    min_default_dcf = min(c_miss * p_target, c_fa * (1 - p_target))
    return float(np.min(dcf) / min_default_dcf)


# 추가할 함수 2: FPR/FNR at EER threshold
def compute_fpr_fnr_at_eer(labels: np.ndarray, scores: np.ndarray) -> dict:
    """EER 임계값에서의 FPR, FNR 분리 반환.
    
    임동훈(2024) Table 5 비교용:
    - FPR: 실제 음성(real)을 fake로 오판하는 비율
    - FNR: 가짜 음성(fake)을 real로 오판하는 비율
    """
    fpr_arr, tpr_arr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr_arr = 1 - tpr_arr
    # EER threshold 찾기
    eer_idx = np.argmin(np.abs(fpr_arr - fnr_arr))
    eer_threshold = thresholds[min(eer_idx, len(thresholds)-1)]
    
    preds = (scores >= eer_threshold).astype(int)
    # label=1 (real) 중 fake로 예측된 비율 → FPR
    real_mask = labels == 1
    fake_mask = labels == 0
    fpr = 1 - preds[real_mask].mean()  # real인데 0(fake)으로 예측
    fnr = preds[fake_mask].mean()       # fake인데 1(real)로 예측
    
    return {
        "fpr": float(fpr),
        "fnr": float(fnr),
        "eer_threshold": float(eer_threshold),
    }
```

**주의**: label 방향 확인 필요. 현재 코드에서 `LABEL_MAP = {"real": 1, "fake": 0}`. `roc_curve(labels, scores, pos_label=1)` → real이 positive. FPR/FNR 계산 시 방향 일치시켜야 함.

**compute_all 수정** (최종: 11개 지표):
```python
def compute_all(labels, scores) -> dict:
    # 기존 6개 + minDCF + FPR/FNR(0.5) + FPR/FNR(EER) = 11개
    preds = (scores >= 0.5).astype(int)
    fpr_fnr_05 = compute_fpr_fnr(labels, scores, threshold=0.5)
    fpr_fnr_eer = compute_fpr_fnr_at_eer(labels, scores)
    return {
        "eer": compute_eer(labels, scores),
        "mindcf": compute_mindcf(labels, scores),
        "accuracy": compute_accuracy(labels, preds),
        "f1": float(f1_score(labels, preds)),
        "precision": float(precision_score(labels, preds)),
        "recall": float(recall_score(labels, preds)),
        "auc": float(roc_auc_score(labels, scores)),
        "fpr": fpr_fnr_05["fpr"],         # threshold=0.5 (실용적 배포)
        "fnr": fpr_fnr_05["fnr"],         # threshold=0.5
        "fpr_eer": fpr_fnr_eer["fpr"],    # EER threshold (학술 비교)
        "fnr_eer": fpr_fnr_eer["fnr"],    # EER threshold
    }
```

### 4-2. build_results.py로 일괄 재평가

**변경 없음** — `build_results.py`는 이미 `compute_all()`을 호출하므로, metrics.py만 확장하면 자동으로 minDCF/FPR/FNR이 results.csv에 포함됨.

**실행**:
```bash
python scripts/03_train/build_results.py --all --output experiments/results_v2.csv
```

13개 실험 × 3 테스트셋 = 39개 minDCF 값 + C_all 4개 × T2 FPR/FNR 자동 산출.

### 4-3. 임동훈 비교표 생성 스크립트 (신규)

**파일**: `scripts/04_analyze/compare_lim.py` (신규)

임동훈(2024) Table 4, 5의 수치를 하드코딩하고, 우리 결과와 대조표를 출력/CSV 저장하는 스크립트.

```python
"""임동훈(2024) 직접 비교표 생성.

사용법:
    python scripts/04_analyze/compare_lim.py --results experiments/results_v2.csv
"""

# 임동훈(2024) Table 4 수치 (KoAAD 테스트)
LIM_TABLE4 = {
    "LCNN-LSTM":          {"eer": 50.85, "accuracy": 52.46, "f1": 42.09, "mindcf": ...},
    "Whisper-LCNN":       {"eer": 66.36, "accuracy": 47.42, "f1": 25.58, "mindcf": ...},
    "SpecRNet":           {"eer": 57.42, "accuracy": 51.28, "f1": 47.07, "mindcf": ...},
    "HPF+Mean+Enhance":   {"eer": 54.55, "accuracy": 52.69, "f1": 42.70, "mindcf": ...},
}

# 임동훈(2024) Table 5 수치 (KoAAD FPR)
LIM_TABLE5 = {
    "LCNN-LSTM":          {"fpr": 65.36},
    "Whisper-LCNN":       {"fpr": 93.47},
    "SpecRNet":           {"fpr": 68.38},
    "HPF+Mean+Enhance":   {"fpr": 78.04},
}

# 우리 results_v2.csv에서 T2(KoAAD) 결과 로드 → 대조표 출력
```

### Phase 4 작업량: ~2시간

| 작업 | 파일 | 예상 시간 |
|------|------|----------|
| minDCF 함수 | metrics.py | 20분 |
| FPR/FNR 함수 | metrics.py | 20분 |
| compute_all 확장 | metrics.py | 10분 |
| 단위 테스트 | test_metrics.py | 20분 |
| 일괄 재평가 실행 | build_results.py --all | 30분 (GPU) |
| 임동훈 비교 스크립트 | compare_lim.py | 20분 |

---

## Phase 5: 강건성 실험

### 5-A. 노이즈 유틸리티 모듈 (신규)

**파일**: `scripts/03_train/augment.py` (신규)

테스트 시/학습 시 공용으로 사용하는 노이즈 증강 함수들.

```python
"""오디오 증강 유틸리티.

기능:
  1. Gaussian noise 추가 (임동훈 재현)
  2. MUSAN noise mixing (SNR 기반)
  3. Codec 압축 시뮬레이션
  4. 조합 증강
"""

import io
import numpy as np
import soundfile as sf
from pathlib import Path
# av(PyAV)는 apply_codec 내부에서 lazy import


class MUSANNoisePool:
    """MUSAN noise 파일을 로드하여 풀로 관리."""
    
    def __init__(self, musan_dir: str, category: str = "noise", sr: int = 16000):
        self.files = sorted(Path(musan_dir).rglob(f"{category}/**/*.wav"))
        self.sr = sr
        # 사전 로드 또는 lazy 로드 선택
    
    def sample(self, length: int) -> np.ndarray:
        """랜덤 noise 파일에서 length 샘플만큼 추출."""
        ...


def add_gaussian_noise(wav: np.ndarray, alpha: float = 0.001) -> np.ndarray:
    """임동훈(2024) Table 3 재현: wav + alpha * N(0,1)"""
    noise = np.random.randn(len(wav)).astype(wav.dtype)
    return wav + alpha * noise


def add_musan_noise(wav: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """SNR 기반 MUSAN 노이즈 혼합.
    
    snr_db: 목표 SNR (dB). 높을수록 깨끗.
    공식: mixed = wav + noise * (rms_wav / rms_noise) * 10^(-snr_db/20)
    """
    rms_wav = np.sqrt(np.mean(wav ** 2) + 1e-8)
    rms_noise = np.sqrt(np.mean(noise ** 2) + 1e-8)
    scale = rms_wav / rms_noise * (10 ** (-snr_db / 20))
    return wav + noise * scale


def apply_codec(wav, sr=16000, codec="mp3", bitrate="64k"):
    """PyAV in-memory 코덱 압축/복원 시뮬레이션.

    wav → (메모리) mp3 encode → decode → wav. subprocess/임시파일 없음 →
    빠르고(학습 중 on-the-fly 적합), Windows 안전, ffmpeg 버전 비의존
    (PyAV가 ffmpeg 번들).
    """
    import av  # lazy import (codec 쓸 때만 — gaussian/musan 경로엔 불필요)
    # 1) encode: AudioFrame(wav, fltp, mono) → libmp3lame(bit_rate) → BytesIO
    # 2) decode: BytesIO → AudioResampler(fltp, mono, sr) → ndarray
    # 3) 출력 길이를 입력에 맞춰 반환 (mp3 priming 지연은 무시 가능)
```

> **주의**: 초기 설계는 ffmpeg subprocess+임시파일 방식이었으나, (1) 샘플마다 프로세스
> spawn으로 학습 병목, (2) Windows `NamedTemporaryFile` 잠금 버그, (3) 시스템 ffmpeg
> 버전 의존 문제로 **PyAV in-memory 방식으로 재작성**했다 (`requirements.txt`에 `av` 추가).

### 5-B. dataset.py 확장 — 학습 시 증강 지원

**파일**: `scripts/03_train/dataset.py`
**변경**: AudioDataset에 증강 옵션 추가

```python
class AudioDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        num_samples=64000,
        is_train=True,
        seed=42,
        # ── 새로 추가 ──
        augment_config: dict | None = None,
        # augment_config 예시:
        # {
        #   "musan_dir": "data/musan/musan",
        #   "musan_snr_range": [10, 20],   # SNR dB 범위
        #   "musan_prob": 0.5,             # 적용 확률
        #   "codec": "mp3",
        #   "codec_bitrate": "64k",
        #   "codec_prob": 0.3,
        # }
    ):
        ...
        self.augment_config = augment_config
        if augment_config and augment_config.get("musan_dir"):
            from augment import MUSANNoisePool
            self.noise_pool = MUSANNoisePool(augment_config["musan_dir"])
        else:
            self.noise_pool = None
    
    def __getitem__(self, idx):
        # ... 기존 wav 로드 + crop/pad ...
        
        # 학습 시에만 증강 적용
        if self.is_train and self.augment_config:
            wav = self._apply_augmentation(wav)  # numpy array 상태에서 적용
        
        return torch.from_numpy(wav), label
    
    def _apply_augmentation(self, wav: np.ndarray) -> np.ndarray:
        """설정에 따라 확률적으로 증강 적용."""
        cfg = self.augment_config
        
        if self.noise_pool and self.rng.random() < cfg.get("musan_prob", 0.5):
            snr = self.rng.uniform(*cfg["musan_snr_range"])
            noise = self.noise_pool.sample(len(wav))
            wav = add_musan_noise(wav, noise, snr)
        
        if cfg.get("codec") and self.rng.random() < cfg.get("codec_prob", 0.3):
            wav = apply_codec(wav, codec=cfg["codec"], bitrate=cfg["codec_bitrate"])
        
        return wav
```

### 5-C. YAML config에 증강 옵션 추가

**변경**: train.py에서 config의 `augmentation` 섹션을 읽어 AudioDataset에 전달

```python
# train.py 수정: AudioDataset 생성 부분
augment_config = cfg.get("augmentation", None)  # None이면 증강 없음
train_ds = AudioDataset(
    cfg["data"]["train_manifest"], num_samples, is_train=True, seed=seed,
    augment_config=augment_config,  # 추가
)
```

### 5-D. 테스트 시 노이즈 평가 스크립트 (신규)

**파일**: `scripts/03_train/eval_noise.py` (신규)

기존 체크포인트를 로드하고, 테스트 데이터에 노이즈를 추가하여 평가. 실험 A 전체 수행.

```python
"""테스트 시 노이즈 강건성 평가.

기존 C_all 체크포인트(specrnet_call, lcnn_call)를 로드하고,
테스트 데이터에 다양한 노이즈를 추가하여 평가한다.

사용법:
    # 전체 실행 (5조건 × 2모델 × 3테스트셋)
    python scripts/03_train/eval_noise.py --musan-dir data/musan/musan

    # 특정 조건만
    python scripts/03_train/eval_noise.py --conditions clean gaussian --models specrnet

    # 특정 테스트셋만
    python scripts/03_train/eval_noise.py --eval-sets T1 T2
"""

NOISE_CONDITIONS = {
    "clean":        {"type": "none"},
    "gaussian_001": {"type": "gaussian", "alpha": 0.001},
    "musan_20db":   {"type": "musan", "snr_db": 20},
    "musan_10db":   {"type": "musan", "snr_db": 10},
    "musan_5db":    {"type": "musan", "snr_db": 5},
}

MODELS = {
    "specrnet": "scripts/03_train/configs/specrnet_call.yaml",
    "lcnn":     "scripts/03_train/configs/lcnn_call.yaml",
}

# 핵심 로직:
# 1. 모델 + 체크포인트 로드
# 2. 테스트셋 로드 (T1/T2/T3)
# 3. 각 조건별로:
#    - wav 로드 → noise 추가 → model forward → scores 수집
#    - compute_all(labels, scores) → 결과 저장
# 4. 결과 CSV 출력: experiments/noise_robustness.csv

class NoisyAudioDataset(AudioDataset):
    """테스트 시 노이즈 추가용 데이터셋.
    
    AudioDataset을 상속하되, __getitem__에서 지정된 노이즈를 적용.
    """
    def __init__(self, manifest_path, num_samples, noise_config, musan_pool=None, seed=42):
        super().__init__(manifest_path, num_samples, is_train=False, seed=seed)
        self.noise_config = noise_config
        self.musan_pool = musan_pool
    
    def __getitem__(self, idx):
        wav_tensor, label = super().__getitem__(idx)
        wav = wav_tensor.numpy()
        
        if self.noise_config["type"] == "gaussian":
            wav = add_gaussian_noise(wav, self.noise_config["alpha"])
        elif self.noise_config["type"] == "musan":
            noise = self.musan_pool.sample(len(wav))
            wav = add_musan_noise(wav, noise, self.noise_config["snr_db"])
        
        return torch.from_numpy(wav), label
```

### 5-E. 신규 YAML 설정 파일들

**실험 B (학습 시 증강)**: 6개 신규 config

```
configs/
├── specrnet_call_aug_musan.yaml     # B-2: SpecRNet + MUSAN
├── lcnn_call_aug_musan.yaml         # B-2: LCNN + MUSAN
├── specrnet_call_aug_codec.yaml     # B-3: SpecRNet + Codec
├── lcnn_call_aug_codec.yaml         # B-3: LCNN + Codec
├── specrnet_call_aug_both.yaml      # B-4: SpecRNet + MUSAN+Codec
└── lcnn_call_aug_both.yaml          # B-4: LCNN + MUSAN+Codec
```

YAML 예시 (`specrnet_call_aug_musan.yaml`):
```yaml
experiment: specrnet_call_aug_musan
data_combo: C_all

model:
  name: specrnet

data:
  train_manifest: manifests/C_all/train.csv
  dev_manifest: manifests/C_all/dev.csv
  eval_manifests:
    T1: manifests/T1_speechfake_ko_test.csv
    T2: manifests/T2_koaad_holdout.csv
    T3: manifests/T3_mlaad_test.csv
  num_samples: 64000
  num_workers: 4

training:
  optimizer: adam
  lr: 0.0001
  batch_size: 64
  epochs: 50
  early_stopping_patience: 10
  weight_decay: 0.0001
  class_weights: [1.0, 1.0]

# ── 증강 설정 (신규) ──
augmentation:
  musan_dir: data/musan/musan
  musan_snr_range: [10, 20]
  musan_prob: 0.5

seed: 42
output_root: experiments
```

**실험 C (발화 길이)**: 4개 신규 config

```
configs/
├── specrnet_call_2sec.yaml          # C-1: SpecRNet 2초
├── lcnn_call_2sec.yaml              # C-1: LCNN 2초
├── specrnet_call_8sec.yaml          # C-3: SpecRNet 8초
└── lcnn_call_8sec.yaml              # C-3: LCNN 8초
```

YAML 차이점: `data.num_samples` 값만 변경
```yaml
# 2초: num_samples: 32000
# 4초: num_samples: 64000 (기존)
# 8초: num_samples: 128000
```

### 5-F. 실험 D: SSL 베이스라인 (선택)

**파일**: `scripts/03_train/models/xlsr_head.py` (신규)

```python
"""XLS-R frozen feature extractor + lightweight head.

XLS-R (300M) 의 feature를 추출하여 저장한 후,
그 위에 경량 head를 학습시킨다.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class XLSRFeatureExtractor:
    """XLS-R feature를 사전 추출하여 .pt로 저장."""
    def __init__(self, model_name="facebook/wav2vec2-xls-r-300m"):
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False


class XLSRHead(nn.Module):
    """XLS-R feature 위의 경량 분류 head."""
    def __init__(self, input_dim=1024, num_classes=2, head_type="aasist"):
        super().__init__()
        if head_type == "fc":
            self.head = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes),
            )
        # head_type == "aasist": AASIST의 backend 부분만 사용
```

**주의**: transformers 라이브러리 필요, GPU 메모리 고려 (feature 사전 추출 방식 권장).

### Phase 5 작업량 요약

| 작업 | 파일 | 예상 시간 | 우선순위 |
|------|------|----------|---------|
| augment.py 작성 | augment.py (신규) | 1시간 | **필수** |
| dataset.py 증강 지원 | dataset.py | 30분 | **필수** |
| train.py 증강 config 연동 | train.py | 15분 | **필수** |
| eval_noise.py 작성 | eval_noise.py (신규) | 1시간 | **필수** |
| MUSAN 다운로드 실행 | download_musan.py | 30분 (다운로드) | **필수** |
| 실험 A 실행 | eval_noise.py | 1시간 (GPU) | **필수** |
| 증강 YAML 6개 생성 | configs/ | 30분 | **필수** |
| 실험 B 실행 (6개 학습) | train.py | 12~18시간 (GPU) | **필수** |
| 길이 YAML 4개 생성 | configs/ | 15분 | 2순위 |
| 실험 C 실행 (4개 학습) | train.py | 8~12시간 (GPU) | 2순위 |
| SSL 모듈 | xlsr_head.py (신규) | 2시간 | 3순위 |
| 실험 D 실행 | train.py | 4~6시간 (GPU) | 3순위 |

---

## Phase 6: 분석 및 시각화

### 6-1. 통합 시각화 스크립트 (신규)

**파일**: `scripts/04_analyze/visualize.py` (신규)

```python
"""논문용 그래프 및 테이블 생성.

사용법:
    python scripts/04_analyze/visualize.py --results experiments/results_v2.csv
    python scripts/04_analyze/visualize.py --noise experiments/noise_robustness.csv
    python scripts/04_analyze/visualize.py --all
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트
matplotlib.rcParams['axes.unicode_minus'] = False

# ── G1: 데이터 커버리지별 EER (Grouped Bar Chart) ──
def plot_coverage_ablation(results_df, output_path):
    """
    X축: 데이터 조합 (C1, C2, C3, C_all)
    Y축: EER (%)
    그룹: 모델별 (AASIST, LCNN, RawNet2 + SpecRNet for C_all)
    테스트셋별 3개 서브플롯 (T1, T2, T3)
    """
    ...

# ── G2: 성능-파라미터 트레이드오프 (Scatter Plot) ──
def plot_efficiency_tradeoff(results_df, output_path):
    """
    X축: 파라미터 수 (log scale)
    Y축: EER (%)
    버블 크기: 추론 속도 (ms)
    레이블: 모델 이름
    C_all 결과만 사용
    """
    ...

# ── G3: SNR별 성능 하락 (Line Chart) ──
def plot_noise_degradation(noise_df, output_path):
    """
    X축: SNR (clean, 20dB, 10dB, 5dB) — 왼쪽이 깨끗
    Y축: EER (%)
    라인: SpecRNet, LCNN (+ 증강 유/무 변형)
    테스트셋별 3개 서브플롯 또는 T2만
    """
    ...

# ── G4: Phase 1→2 개선 (Before/After Bar Chart) ──
def plot_improvement(results_df, output_path):
    """
    모델별 최악 EER (Phase 1) vs C_all EER (Phase 2)
    강조: LCNN T3 81.30% → 0.02% (4,065x 개선)
    """
    ...

# ── 테이블 생성 (LaTeX/Markdown) ──
def generate_tables(results_df, noise_df, output_dir):
    """T1~T6 논문용 테이블 생성.
    
    포맷: LaTeX tabular + Markdown (둘 다)
    """
    ...
```

### 6-2. 출력 파일 구조

```
experiments/
├── results_v2.csv              # Phase 4: minDCF/FPR/FNR 포함
├── noise_robustness.csv        # Phase 5-A: 노이즈 강건성 결과
├── augmentation_results.csv    # Phase 5-B: 증강 학습 결과
├── length_results.csv          # Phase 5-C: 발화 길이 결과
├── figures/                    # Phase 6: 그래프
│   ├── G1_coverage_ablation.pdf
│   ├── G2_efficiency_tradeoff.pdf
│   ├── G3_noise_degradation.pdf
│   └── G4_improvement.pdf
├── tables/                     # Phase 6: 테이블
│   ├── T1_full_results.tex
│   ├── T2_call_comparison.tex
│   ├── T3_lim_comparison.tex
│   ├── T4_noise_robustness.tex
│   ├── T5_augmentation.tex
│   └── T6_asvspoof_indirect.tex
└── lim_comparison.csv          # Phase 4: 임동훈 비교표
```

---

## 구현 순서 및 의존성

```
Step 1: metrics.py 확장 (minDCF, FPR/FNR)
   │    ↳ 의존: 없음. 단독 수행 가능.
   │
Step 2: build_results.py --all 재실행 → results_v2.csv
   │    ↳ 의존: Step 1 + GPU 서버 (기존 체크포인트 필요)
   │
Step 3: compare_lim.py 작성 → lim_comparison.csv
   │    ↳ 의존: Step 2
   │
Step 4: augment.py 작성 (노이즈/코덱 유틸리티)
   │    ↳ 의존: 없음. 로컬에서 작성 가능.
   │
Step 5: MUSAN 다운로드
   │    ↳ 의존: GPU 서버에서 실행. download_musan.py 이미 있음.
   │
Step 6: eval_noise.py 작성 + 실행 → noise_robustness.csv
   │    ↳ 의존: Step 4 + Step 5 + GPU 서버
   │
Step 7: dataset.py 증강 지원 추가 + train.py 연동
   │    ↳ 의존: Step 4
   │
Step 8: 증강 YAML 6개 생성
   │    ↳ 의존: Step 7
   │
Step 9: 실험 B 실행 (6개 학습) → augmentation_results.csv
   │    ↳ 의존: Step 8 + GPU 서버 (12~18시간)
   │
Step 10: 길이 YAML 4개 생성 + 실험 C 실행 → length_results.csv
   │     ↳ 의존: Step 7 + GPU 서버 (8~12시간)
   │     ↳ Step 9과 병렬 가능 (서로 독립)
   │
Step 11: visualize.py 작성 + 실행 → figures/ + tables/
   │     ↳ 의존: Step 2, 6, 9, 10 (모든 결과 CSV)
   │
Step 12 (선택): SSL 베이스라인
         ↳ 의존: Step 7 + transformers 라이브러리
```

**병렬화 가능 구간**:
- Step 1~4: 로컬에서 코드 작성 (GPU 불필요)
- Step 5: MUSAN 다운로드 (Step 4와 병렬)
- Step 9, 10: 증강/길이 학습 (서로 독립, 순차 또는 멀티 GPU시 병렬)

---

## 파일 변경 요약

| 파일 | 변경 유형 | 상태 | 내용 |
|------|----------|:----:|------|
| `scripts/03_train/metrics.py` | **수정** | ✅ | minDCF, FPR/FNR, FPR/FNR@EER 함수 추가, compute_all 확장 (11개 지표) |
| `scripts/03_train/dataset.py` | **수정** | ✅ | augment_config 파라미터, _apply_augmentation 메서드 |
| `scripts/03_train/train.py` | **수정** | ✅ | augmentation config 연동, TensorBoard에 minDCF/FPR/FNR 로깅, 평가 출력 포맷 |
| `scripts/03_train/augment.py` | **신규** | ✅ | MUSANNoisePool, add_gaussian_noise, add_noise_snr, apply_codec |
| `scripts/03_train/eval_noise.py` | **신규** | ✅ | NoisyAudioDataset, 5조건×2모델 일괄 평가 → noise_robustness.csv |
| `scripts/04_analyze/compare_lim.py` | **신규** | ✅ | 임동훈(2024) Table 4,5 대조표 생성 → lim_comparison.csv |
| `scripts/04_analyze/visualize.py` | **신규** | ✅ | G1~G4 그래프 (PDF+PNG), T1~T4 테이블 (Markdown) |
| `configs/specrnet_call_aug_musan.yaml` | **신규** | ✅ | B-2: SpecRNet MUSAN 증강 |
| `configs/lcnn_call_aug_musan.yaml` | **신규** | ✅ | B-2: LCNN MUSAN 증강 |
| `configs/specrnet_call_aug_codec.yaml` | **신규** | ✅ | B-3: SpecRNet Codec 증강 |
| `configs/lcnn_call_aug_codec.yaml` | **신규** | ✅ | B-3: LCNN Codec 증강 |
| `configs/specrnet_call_aug_both.yaml` | **신규** | ✅ | B-4: SpecRNet MUSAN+Codec |
| `configs/lcnn_call_aug_both.yaml` | **신규** | ✅ | B-4: LCNN MUSAN+Codec |
| `configs/specrnet_call_2sec.yaml` | **신규** | ✅ | C-1: SpecRNet 2초 |
| `configs/lcnn_call_2sec.yaml` | **신규** | ✅ | C-1: LCNN 2초 |
| `configs/specrnet_call_8sec.yaml` | **신규** | ✅ | C-3: SpecRNet 8초 |
| `configs/lcnn_call_8sec.yaml` | **신규** | ✅ | C-3: LCNN 8초 |
| `scripts/03_train/test_metrics.py` | **신규** | ✅ | metrics.py 단위 테스트 20개 (EER, minDCF, FPR/FNR, compute_all) |
| `scripts/03_train/models/xlsr_head.py` | **신규 (선택)** | 미구현 | SSL 베이스라인 (3순위) |

**기존 코드 변경 최소화**: 4개 파일만 수정 (metrics.py, dataset.py, train.py, build_results.py). 나머지는 전부 신규 파일.

---

## 코드 검토 결과 (2026-06-05)

### 수정 완료

| # | 문제 | 파일 | 수정 |
|---|------|------|------|
| 1 | build_results.py 콘솔 출력에 minDCF/FPR/FNR 누락 | build_results.py L109 | 포맷 통일 |
| 2 | eval_noise.py에 evaluate() 함수 중복 복사 | eval_noise.py L104-123 | `from train import evaluate` 로 변경 |
| 3 | compare_lim.py accuracy/f1 단위 불일치 (소수 vs %) | compare_lim.py L86-96 | 전부 % 변환 통일 |
| 4 | build_results.py에서 평가 에러 시 전체 크래시 | build_results.py main() | try/except + traceback 추가 |
| 5 | build_results.py CSV fieldnames 불일치 가능 | build_results.py main() | 키 합집합 방식으로 변경 |

### 추가 수정 완료

| # | 문제 | 파일 | 수정 |
|---|------|------|------|
| 6 | FPR/FNR: EER threshold 버전 누락 | metrics.py | `fpr_eer`, `fnr_eer` 추가 (compute_all 11개 지표로 확장) |
| 7 | test_metrics.py 미작성 | test_metrics.py (신규) | 20개 단위 테스트 작성, 전부 통과 ✅ |
| 8 | train.py/build_results.py에 fpr_eer/fnr_eer 로깅 누락 | train.py, build_results.py | TB 로깅 + 콘솔 출력 추가 |

### 미해결 (참고)

| # | 항목 | 상태 | 비고 |
|---|------|------|------|
| 1 | augment.py 리샘플링 조잡 | 사소 | MUSAN이 16kHz면 발동 안 함 |
| 2 | xlsr_head.py (SSL) | 3순위 | 시간 여유 시 구현 |

---

## 후속 검토 및 수정 (2026-06-05, 2차)

1차 검토 이후 추가 코드 리뷰에서 발견·수정한 사항. 전부 검증 완료(metrics 단위테스트 20/20,
체크포인트 로딩·노이즈 재현성·PyAV codec 동작 확인).

| # | 심각도 | 문제 | 파일 | 수정 |
|---|--------|------|------|------|
| 1 | 🔴 치명 | `torch.compile`로 저장된 체크포인트(키에 `_orig_mod.` 접두사)를 raw 모델에 로드 시 실패 → Phase 4 재평가·Phase 5-A 노이즈 평가가 GPU에서 전부 크래시 | train.py, build_results.py, eval_noise.py | `load_checkpoint()` 헬퍼 추가(접두사 자동 제거), 두 로드 지점에 적용 |
| 2 | 🟠 방법론 | eval_noise 노이즈가 비결정적(gaussian 전역 RNG) + 모델 간 다른 노이즈 → 강건성 표 재현·페어링 불가 | eval_noise.py, augment.py | `NoisyAudioDataset`이 `default_rng(seed+idx)` 사용, `add_gaussian_noise`/`MUSANNoisePool.sample`에 `rng` 인자 추가 → 인덱스별 결정적, 모델 간 동일 입력 |
| 3 | 🟠 실용 | codec 증강이 샘플마다 ffmpeg subprocess+임시파일 → 학습 병목 + Windows 임시파일 잠금 버그 | augment.py, requirements.txt | `apply_codec` PyAV in-memory 재작성(`_parse_bitrate` 추가), `av` 의존성 추가 |
| 4 | 🟡 | `train.py --all`이 신규 config 혼재 시 완료된 13개까지 재학습+results.csv 중복 append | train.py | `--all`이 완료 실험 기본 skip, `--force`로 강제 재학습 |
| 5 | 🟡 | `build_results.py` 기본 출력이 results.csv → 학습 로그 덮어쓸 위험 | build_results.py | 기본 출력을 `results_v2.csv`로 변경 |
| 6 | 🟡 문서 | "Weighted BCE" 표기가 실제 CrossEntropyLoss(미가중)와 불일치 | 연구제안서_v3.md, RESEARCH_PHASES.md | "CrossEntropyLoss (class_weights [1.0,1.0])"로 정정 |
| 7 | 🟡 | gaussian α=0.001 near-clean 미주석 / MUSAN noise 카테고리 발견성 / visualize G1 dead code·NaN 막대 | eval_noise.py, dataset.py, visualize.py | 주석 추가(near-clean, `musan_category`), G1 dead code 제거·NaN 막대 미표시 |

신규 헬퍼/함수: `train.load_checkpoint`, `augment._parse_bitrate`. 신규 의존성: `av`(PyAV).

---

## GPU 실행 시간 추정

| 단계 | 내용 | GPU 시간 |
|------|------|---------|
| Phase 4 재평가 | 13개 체크포인트 × 3 테스트셋 (학습 없음) | ~30분 |
| Phase 5-A | 5조건 × 2모델 × 3 테스트셋 (학습 없음) | ~1시간 |
| Phase 5-B | 6개 신규 학습 (C_all 규모) | ~12~18시간 |
| Phase 5-C | 4개 신규 학습 | ~8~12시간 |
| Phase 5-D | 2개 SSL (선택) | ~4~6시간 |
| **합계** | | **~25~37시간** |

---

*이 문서는 코드 구현 과정에서 업데이트합니다.*
