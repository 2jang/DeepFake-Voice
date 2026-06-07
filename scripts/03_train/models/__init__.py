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


def get_model(name: str, num_classes: int = 2):
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes)
