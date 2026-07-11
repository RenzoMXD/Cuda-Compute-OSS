"""CPU-only tests for --data-rank 0 being honored, not silently discarded.

Run:  python strategy/tests/test_data_rank_zero.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy import storage


def test_lowrank_data_rank_zero_is_honored():
    # Before the fix, `data_rank or default` treated 0 as falsy and silently
    # substituted max(1, n // 32), so this generated a much higher-rank
    # matrix than requested with no error or warning.
    mat_zero = storage.generate(64, np.float32, False, None, seed=1,
                                 fill="lowrank", data_rank=0)
    mat_default = storage.generate(64, np.float32, False, None, seed=1,
                                    fill="lowrank", data_rank=None)
    assert np.all(mat_zero == 0.0)
    assert not np.array_equal(mat_zero, mat_default)


def test_decaying_spectrum_data_rank_zero_is_honored():
    mat_zero = storage.generate(64, np.float32, False, None, seed=1,
                                 fill="decaying-spectrum", data_rank=0)
    mat_default = storage.generate(64, np.float32, False, None, seed=1,
                                    fill="decaying-spectrum", data_rank=None)
    assert np.all(mat_zero == 0.0)
    assert not np.array_equal(mat_zero, mat_default)


def test_data_rank_zero_raises_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        storage.generate(32, np.float32, False, None, seed=1,
                          fill="lowrank", data_rank=0)
        storage.generate(32, np.float32, False, None, seed=1,
                          fill="decaying-spectrum", data_rank=0)


def test_data_rank_none_still_uses_default():
    # data_rank=None (the CLI's "unset" sentinel) must still fall back to
    # max(1, n // 32), unlike data_rank=0.
    mat_default = storage.generate(64, np.float32, False, None, seed=1,
                                    fill="lowrank", data_rank=None)
    mat_explicit = storage.generate(64, np.float32, False, None, seed=1,
                                     fill="lowrank", data_rank=max(1, 64 // 32))
    assert np.array_equal(mat_default, mat_explicit)


if __name__ == "__main__":
    try:
        import pytest
    except ImportError:
        print("SKIP  strategy/tests/test_data_rank_zero.py (pytest required)")
        sys.exit(0)

    raise SystemExit(pytest.main([__file__, "-v"]))
