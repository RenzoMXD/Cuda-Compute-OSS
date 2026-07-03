"""Tests for the standalone subspace strategy and its transforms (GPU/PyTorch).

CCO computes on the GPU only, so these need a CUDA/MPS device; they skip cleanly
when none is present.

Run:  python strategy/tests/test_subspace.py   (or via pytest)
"""
import os
import sys

import numpy as np

# make the project root importable so `import strategy` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy import subspace_matmul, Config, register_transform, available
from strategy.backend import Backend
from strategy import subspace
from strategy.transforms import Transform, get_transform


def _gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available()
                    or (getattr(torch.backends, "mps", None)
                        and torch.backends.mps.is_available()))
    except Exception:  # noqa: BLE001
        return False


HAVE_GPU = _gpu_available()
BK = Backend(verbose=False) if HAVE_GPU else None


def _rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) or 1.0))


def _run(A, B, m, transform, dtype="fp64"):
    cfg = Config(dtype=dtype, rank_m=m, transform=transform, verbose=False)
    C = np.zeros_like(A)
    subspace.multiply_subspace(A, B, C, BK, cfg)
    return C


# -- streaming primitives -------------------------------------------------
def test_compress_matches_direct():
    rng = np.random.default_rng(0)
    n, m = 40, 12
    X = rng.standard_normal((n, n))
    Q = np.linalg.qr(rng.standard_normal((n, m)))[0]
    got = np.asarray(subspace.compress(X, Q, BK, np.float64))
    assert _rel(got, Q.T @ X @ Q) < 1e-12


def test_reconstruct_matches_direct():
    rng = np.random.default_rng(1)
    n, m = 40, 12
    Ctil = rng.standard_normal((m, m))
    Q = np.linalg.qr(rng.standard_normal((n, m)))[0]
    C = np.zeros((n, n))
    subspace.reconstruct(Ctil, Q, C, BK, np.float64)
    assert _rel(C, Q @ Ctil @ Q.T) < 1e-12


def test_stream_primitives():
    rng = np.random.default_rng(2)
    n, m = 32, 7
    X = rng.standard_normal((n, n))
    Om = rng.standard_normal((n, m))
    assert _rel(np.asarray(subspace.stream_gemm_right(X, Om, BK, np.float64)), X @ Om) < 1e-12
    assert _rel(np.asarray(subspace.stream_gemm_left_t(X, Om, BK, np.float64)), X.T @ Om) < 1e-12


def test_exact_baseline_matches_numpy():
    rng = np.random.default_rng(3)
    n = 48
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    C = np.zeros((n, n))
    subspace.multiply_exact(A, B, C, BK, Config(dtype="fp64", verbose=False))
    assert _rel(C, A @ B) < 1e-12


# -- exactness / accuracy properties --------------------------------------
def test_exact_when_m_equals_n():
    # M = N => Q spans R^N => P = I => exact. rsvd's four sketches concatenated
    # to N columns span the full space for full-rank A/B.
    rng = np.random.default_rng(4)
    n = 24
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    assert _rel(_run(A, B, n, "rsvd"), A @ B) < 1e-9


def test_custom_transform_exact_with_shared_basis():
    # A, B share an eigenbasis Q0 (rank r); a transform that returns Q0 with
    # M = r reconstructs exactly — the mechanism a contributor plugs into.
    rng = np.random.default_rng(5)
    n, r = 50, 6
    Q0 = np.linalg.qr(rng.standard_normal((n, r)))[0]
    A = Q0 @ np.diag(rng.standard_normal(r)) @ Q0.T
    B = Q0 @ np.diag(rng.standard_normal(r)) @ Q0.T

    class Shared(Transform):
        name = "shared"
        def basis(self, n, m, backend, dtype, A=None, B=None):
            return backend.to_device(Q0.astype(dtype, copy=False))

    assert _rel(_run(A, B, r, Shared()), A @ B) < 1e-9


def test_rsvd_recovers_low_rank_product():
    rng = np.random.default_rng(6)
    n, r, m = 96, 5, 48                     # m/4 = 12 >= r per captured space
    A = rng.standard_normal((n, r)) @ rng.standard_normal((r, n))
    B = rng.standard_normal((n, r)) @ rng.standard_normal((r, n))
    assert _rel(_run(A, B, m, "rsvd"), A @ B) < 1e-8


def test_rsvd_more_dims_not_worse_on_low_rank():
    # For low-rank data, a larger M can only capture at least as much.
    rng = np.random.default_rng(7)
    n, r = 60, 8
    A = rng.standard_normal((n, r)) @ rng.standard_normal((r, n))
    B = rng.standard_normal((n, r)) @ rng.standard_normal((r, n))
    assert _rel(_run(A, B, 48, "rsvd"), A @ B) <= _rel(_run(A, B, 16, "rsvd"), A @ B) + 1e-9


# -- registry / extensibility --------------------------------------------
def test_registry_custom_transform():
    class IdentityBlock(Transform):
        name = "idblock"

        def basis(self, n, m, backend, dtype, A=None, B=None):
            Q = np.zeros((n, m), dtype=dtype)
            Q[np.arange(m), np.arange(m)] = 1.0
            return backend.to_device(Q)

    register_transform("idblock", IdentityBlock)
    assert "idblock" in available()
    assert isinstance(get_transform("idblock"), IdentityBlock)
    rng = np.random.default_rng(8)
    n = 20
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    assert _rel(_run(A, B, n, "idblock"), A @ B) < 1e-9   # M=N => identity => exact


def test_public_api():
    rng = np.random.default_rng(9)
    n, r = 64, 8
    A = (rng.standard_normal((n, r)) @ rng.standard_normal((r, n))).astype(np.float32)
    B = (rng.standard_normal((n, r)) @ rng.standard_normal((r, n))).astype(np.float32)
    C = subspace_matmul(A, B, config=Config(transform="rsvd", rank_m=40,
                                            verbose=False))
    assert _rel(C, A @ B) < 1e-3


if __name__ == "__main__":
    if not HAVE_GPU:
        print("SKIP  all tests (no CUDA/MPS GPU; CCO computes on GPU only)")
        sys.exit(0)
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
