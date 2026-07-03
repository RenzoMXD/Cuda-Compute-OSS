"""Tests for the eval system.

Metric tests run anywhere (pure NumPy). End-to-end evaluate/scaling tests use
the GPU (PyTorch) and skip when no CUDA/MPS device is present.

    python eval/tests/test_eval.py        (or)   python -m pytest eval/tests -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval import metrics
from eval.evaluator import EvalConfig, evaluate, estimate_scaling


class _Skip(Exception):
    pass


def _gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available()
                    or (getattr(torch.backends, "mps", None)
                        and torch.backends.mps.is_available()))
    except Exception:  # noqa: BLE001
        return False


HAVE_GPU = _gpu_available()


# ---- metrics -------------------------------------------------------------
def test_accuracy_identical_is_one():
    C = np.random.default_rng(0).standard_normal((32, 32))
    assert metrics.accuracy(C, C) == 1.0


def test_accuracy_is_bounded_zero_one():
    rng = np.random.default_rng(1)
    C = rng.standard_normal((32, 32))
    Chat = C + 100.0 * rng.standard_normal((32, 32))   # very wrong
    a = metrics.accuracy(C, Chat)
    assert 0.0 <= a <= 1.0


def test_accuracy_floors_at_zero():
    # Approx with the negated matrix -> error >> 1 -> clamped to 0, never negative.
    C = np.ones((8, 8))
    assert metrics.accuracy(C, -C) == 0.0


def test_score_gated_by_accuracy_floor():
    # Accuracy below the floor -> score forced to 0 regardless of speed/memory.
    gated = metrics.score(0.5, peak_vram_bytes=1e6, latency_s=0.01,
                          accuracy_floor=0.9)
    assert gated == 0.0
    ok = metrics.score(0.95, peak_vram_bytes=1e6, latency_s=0.01,
                       accuracy_floor=0.9)
    assert ok > 0.0


def test_score_no_floor_by_default():
    # Default floor is 0.0 -> even low accuracy is scored (not gated).
    assert metrics.score(0.01, 1e6, 0.01) > 0.0


def test_score_monotonic_in_accuracy():
    lo = metrics.score(0.2, 1e6, 0.01)
    hi = metrics.score(0.9, 1e6, 0.01)
    assert hi > lo


# ---- end-to-end evaluate (GPU) -------------------------------------------
def test_evaluate_smoke():
    if not HAVE_GPU:
        raise _Skip()
    ev = EvalConfig(n=96, pairs=2, dtype="fp32", fill="lowrank", data_rank=4,
                    transforms=["rsvd"], verbose=False)
    out = evaluate(ev)
    assert set(out["transforms"]) == {"rsvd"}
    for r in out["transforms"].values():
        assert 0.0 <= r["accuracy"] <= 1.0
        assert r["latency_s"] > 0.0
        assert r["score"] >= 0.0
    assert out["best"] == "rsvd"


def test_rsvd_accurate_on_lowrank():
    # On genuinely low-rank data the data-aware rsvd basis reconstructs closely.
    if not HAVE_GPU:
        raise _Skip()
    ev = EvalConfig(n=128, pairs=2, dtype="fp32", fill="lowrank", data_rank=6,
                    rank_m=48, transforms=["rsvd"], verbose=False)
    out = evaluate(ev)
    assert out["transforms"]["rsvd"]["accuracy"] > 0.99


def test_scaling_exponent_runs():
    if not HAVE_GPU:
        raise _Skip()
    ev = EvalConfig(dtype="fp32", fill="lowrank", data_rank=4, rank_m=16,
                    transforms=["rsvd"], verbose=False)
    out = estimate_scaling([64, 128, 192], ev)
    assert "fitted_exponent_p" in out
    assert np.isfinite(out["fitted_exponent_p"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = skipped = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except _Skip:
            skipped += 1
            print(f"SKIP  {fn.__name__} (no GPU)")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed - skipped}/{len(fns) - skipped} passed"
          + (f", {skipped} skipped" if skipped else ""))
    sys.exit(1 if failed else 0)
