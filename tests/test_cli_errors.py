"""Invalid CLI arguments must be reported cleanly (exit code 2 + an ``error:``
line on stderr), never surfaced as an uncaught traceback.

Both ``matmul`` and ``strategy`` validate ``--n`` and build/validate their
``Config`` *before* any device work, so every case here is rejected on CPU with
no GPU/PyTorch present -- which is exactly how they run in PR CI.
"""
from __future__ import annotations

import pytest

from matmul import cli as matmul_cli
from strategy import cli as strategy_cli

BAD_ARGS = [
    ["--vram-fraction", "1.5", "--n", "8"],   # vram_fraction > 0.95
    ["--vram-fraction", "0", "--n", "8"],     # vram_fraction <= 0
    ["--n", "0"],                             # non-positive n
    ["--n", "-4"],                            # negative n
]


@pytest.mark.parametrize("main", [matmul_cli.main, strategy_cli.main],
                         ids=["matmul", "strategy"])
@pytest.mark.parametrize("argv", BAD_ARGS, ids=lambda a: " ".join(a))
def test_bad_args_exit_cleanly(main, argv, capsys):
    rc = main(argv)
    assert rc == 2, f"expected exit 2 for {argv}, got {rc}"
    assert "error:" in capsys.readouterr().err


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
