"""Model registry."""

from __future__ import annotations

from .aasist import AASIST
from .lcnn import LCNN
from .rawnet2 import RawNet2
from .specrnet import SpecRNet
from .xlsr_head import XLSRAasist, XLSRfc

MODEL_REGISTRY: dict[str, type] = {
    "aasist": AASIST,
    "lcnn": LCNN,
    "rawnet2": RawNet2,
    "specrnet": SpecRNet,
    "xlsr_aasist": XLSRAasist,
    "xlsr_fc": XLSRfc,
}


def get_model(name: str, num_classes: int = 2, **kwargs):
    """모델 생성. kwargs는 모델별 추가 인자 (예: LCNN의 num_samples)."""
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[name]
    # 모델이 지원하는 인자만 전달
    import inspect
    sig = inspect.signature(cls.__init__)
    valid = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(num_classes=num_classes, **valid)
