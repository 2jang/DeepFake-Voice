#!/usr/bin/env python3
"""Phase 2.2 step 2: 3개 모델 forward/backward pass 검증.

사용법:
    python scripts/03_train/verify_models.py
    python scripts/03_train/verify_models.py --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import get_model

MODELS = ["aasist", "lcnn", "rawnet2"]
NUM_SAMPLES = 64_000
BATCH_SIZE = 4


def verify(name: str, device: torch.device) -> bool:
    print(f"\n  ── {name.upper()} ──")
    try:
        model = get_model(name).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Parameters: {n_params:,} ({n_params * 4 / 1e6:.2f} MB)")

        x = torch.randn(BATCH_SIZE, NUM_SAMPLES, device=device)
        logits = model(x)
        print(f"    Forward:  input {tuple(x.shape)} → output {tuple(logits.shape)}")
        assert logits.shape == (BATCH_SIZE, 2), f"Expected (B, 2), got {logits.shape}"

        loss = nn.CrossEntropyLoss()(logits, torch.randint(0, 2, (BATCH_SIZE,), device=device))
        loss.backward()
        print(f"    Backward: loss={loss.item():.4f}")

        grad_ok = all(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        print(f"    Gradients: {'OK' if grad_ok else 'WARN (some zero grads)'}")
        print(f"    [PASS] {name}")
        return True

    except Exception as e:
        print(f"    [FAIL] {name}: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify model forward/backward")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"  Device: {device}")
    results = {name: verify(name, device) for name in MODELS}

    print(f"\n  ── Summary ──")
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"    {name:10s} {status}")
        if not ok:
            all_pass = False

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
