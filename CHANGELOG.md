# Changelog

All notable changes to CCO are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **`eval` averages over 100 couples by default** (`--pairs` / `EvalConfig.pairs`
  default `3` → `100`) for steadier numbers.

### Removed
- **`random`, `dct`, and `provided` transforms** dropped from `strategy` —
  `rsvd` is now the only built-in transform (and the default). Add your own by
  subclassing `Transform` and calling `register_transform`.

### Changed
- **MSE removed; the score gate is now an accuracy floor.** `eval` no longer
  computes or reports mean-squared error. The score is hard-gated by
  `Accuracy < floor` instead of `MSE > τ`: `EvalConfig.mse_threshold` →
  `accuracy_floor` (default `0.0`), CLI `--mse-threshold` → `--min-accuracy`,
  and `metrics.mse()` is dropped.
- **Reference regime is now `1024 × 1024`, full-rank, on RTX 4060/5060 (8 GB).**
  `--n` defaults to `1024` across the CLIs; `EvalConfig` defaults to `n=1024`,
  full-rank (`fill="random"`). Docs (README, CONTRIBUTING, BENCHMARKS) are
  reframed around this — the honest result being that a subspace of `M ≪ N`
  cannot beat exact on full-rank data (accuracy ≈ 0, gated by the accuracy
  floor); the strategy
  wins only on compressible/low-rank inputs.
- **GPU-only, PyTorch-only compute.** CCO now computes every matrix product on a
  GPU (CUDA or Apple MPS) via PyTorch. The **NumPy CPU backend and the CuPy
  backend are removed**, along with the `--backend`/`--cpu` flags and the
  `Config.backend`/`prefer_gpu` fields — there is nothing to select. `Backend`
  raises a clear error when no GPU is present. NumPy is retained only for
  host-side arrays, memmap storage, and reference/scoring math.
- Per-engine products: the **normal (exact) `matmul/` engine uses `torch.bmm`**,
  the **smart (subspace) `strategy/` engine uses `torch.matmul`** (via
  `Backend.matmul`).
- `eval` peak-VRAM probe now measures GPU memory (CUDA `max_memory_allocated`;
  MPS sampling) instead of host RSS.

### Removed
- The CuPy double-buffered `--overlap` pipeline and `matmul/_blas.py` (a
  NumPy-BLAS thread-tuning helper) — both irrelevant without CPU/CuPy.
- `cupy-cuda12x` from `requirements.txt`; `torch>=2.1` is now required.

### Notes
- Tests run on the GPU and **skip** when no CUDA/MPS device is present.

## [0.1.0] — 2026-07-03

Initial open-source scaffolding.

### Added
- **Normal (exact) engine** (`matmul/`): out-of-core tiled/streamed `O(N³)`
  cuBLAS multiply with a transparent NumPy CPU fallback, `--overlap` pipeline,
  and disk-backed storage for `N` far beyond device memory.
- **Smart (subspace) strategies** (`strategy/`): compress → multiply →
  reconstruct at `O(N²M)`, with a pluggable transform (`rsvd`) — the
  contribution surface.
- **Scorer** (`eval/`): generates random couples, runs normal + smart on
  identical inputs, and reports accuracy, latency, peak VRAM, and time
  complexity, folded into a single accuracy-gated score; empirical `N^p` fit via
  `--sweep`.
- Project docs: `README.md`, `CONTRIBUTING.md`, `BENCHMARKS.md`, `LICENSE` (MIT),
  and a PR scorecard template.

### Performance
- CPU tiled path sizes the BLAS thread pool to the tile, avoiding thread-pool
  oversubscription on small tiles (~150× faster at `T=128`). No-op on GPU.
