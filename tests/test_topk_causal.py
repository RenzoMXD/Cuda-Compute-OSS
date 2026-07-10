"""Regression tests: topk landmark policy must be causal in causal mode (issue #170).

topk selects a single global landmark set from a mean over all queries + a top-k
over all keys, so in causal mode a query's output would depend on future q/k. A
global set cannot be causal, so causal runs fall back to the position-based pooled
selection (causally correct). Non-causal topk is unchanged.

CPU-safe: skips cleanly when torch is not installed.
Run:  python tests/test_topk_causal.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

if torch is not None:
    from attention.hybrid import landmark_global_attention


def _skip():
    if torch is None:
        print("SKIP  torch not installed")
        return True
    return False


def _lga(q, k, v, causal, policy):
    return landmark_global_attention(q, k, v, num_landmarks=4, causal=causal, policy=policy)


def test_topk_causal_is_finite():
    """#133 contract: topk + causal must still produce finite output."""
    if _skip():
        return
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 16, 4) for _ in range(3))
    out = _lga(q, k, v, causal=True, policy="topk")
    assert out.shape == q.shape and torch.isfinite(out).all()


def test_topk_causal_does_not_leak_future():
    """Perturbing a future q/k must not change any earlier query's output."""
    if _skip():
        return
    for which in ("q", "k"):
        for seed in range(20):
            torch.manual_seed(seed)
            q, k, v = (torch.randn(1, 1, 16, 4) for _ in range(3))
            base = _lga(q, k, v, causal=True, policy="topk")
            q2, k2 = q.clone(), k.clone()
            (q2 if which == "q" else k2)[0, 0, 15] += 8.0
            out2 = _lga(q2, k2, v, causal=True, policy="topk")
            worst = max((out2[0, 0, t] - base[0, 0, t]).abs().max().item() for t in range(15))
            assert worst < 1e-5, f"future {which} leaked into a past query (seed {seed}): {worst}"


def test_topk_causal_matches_pooled():
    if _skip():
        return
    torch.manual_seed(3)
    q, k, v = (torch.randn(1, 1, 20, 5) for _ in range(3))
    assert torch.allclose(_lga(q, k, v, True, "topk"), _lga(q, k, v, True, "pooled"), atol=1e-6)


def test_topk_noncausal_is_unchanged():
    """Non-causal topk still uses its own (global) selection, distinct from pooled."""
    if _skip():
        return
    torch.manual_seed(4)
    q, k, v = (torch.randn(1, 1, 20, 5) for _ in range(3))
    topk = _lga(q, k, v, False, "topk")
    pooled = _lga(q, k, v, False, "pooled")
    assert torch.isfinite(topk).all() and not torch.allclose(topk, pooled, atol=1e-4)


if __name__ == "__main__":
    fns = [v for kk, v in sorted(globals().items()) if kk.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
