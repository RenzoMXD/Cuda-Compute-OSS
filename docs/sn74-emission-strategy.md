# SN74 Emission Strategy for CCO

*Reference plan, July 2026. Based on a review of every repository above a 5%
emission share in gittensor's `master_repositories.json` (as of commit
89a7d8c, 2026-07-06). Keep for the future weight-adjustment proposal.*

## How the top repos earn (survey)

| repo | share | review model | scoring economy | key thresholds |
|---|---:|---|---|---|
| sparkinfer | 34.25% | deterministic eval bot on pinned RTX 5090; copycat strikes | `fixed_base_score` 1.0 × `eval:XS–XL` tiers (0.5–4.0); `none`/`REJECT` = 0 | cred ≥ 0.2, 0 min PRs, cut 0.5 |
| metagraphed | 23% | human/agent + hard CI gates; anti-farming ("one subnet = one file = one PR") | token-scored × maintainer labels (`bug` .05 / `feature` .25 / `priority` 1.5 / `slop` 0) | cred ≥ 0.7, 7-day lookback, 3-day decay, cut 0.55 |
| vanguarstew | 10% | human review; `agent/` open, `benchmark/` validator-owned | token-scored × work taxonomy (`core-correctness` 2.0 → `docs` 0.8) | max 2 open PRs, cut 0.5 |
| gittensory | 10% | fully automated one-shot: auto-merge if green (~97% patch coverage) or auto-close; fresh PR to retry | same label economy as metagraphed | cred ≥ 0.7, cut 0.55 |
| kata | 8% | tournament rounds vs reigning "king" in sandbox | winner-take-all: `kata:winner:*` 1.0, all else 0 | 1 merged PR min, cred 0, cut 0.4 |
| entrius/gittensor | 6% | team human review; weights changed by PR (`weight_adjustment` template) | default token scoring × light labels (`enhancement` 1.3 … `refactor` 0.1) | subnet defaults, no cut |

**Patterns that won:** `default_label_multiplier: 0.0` (unlabeled = zero) in 5/6;
`fixed_base_score: 1.0` + tiered labels to pay outcomes not LOC; deterministic
bots as the pricer; lenient entry gates; maintainer cuts 0.4–0.55 funding the
verification infrastructure.

**Patterns to avoid for CCO:** kata's winner-take-all (suits scheduled
tournaments, not a slow-moving frontier) and metagraphed's 3-day decay
(punishes careful research work).

## Target registry entry for CCO

```json
"zeokin/Cuda-Compute-OSS": {
  "emission_share": 0.01,
  "fixed_base_score": 1.0,
  "trusted_label_pipeline": true,
  "default_label_multiplier": 0.0,
  "label_multipliers": {
    "eval:XL": 4.0, "eval:L": 2.5, "eval:M": 1.5,
    "eval:S": 1.0, "eval:XS": 0.5,
    "eval:BASELINE": 1.0, "eval:none": 0.0, "eval:REJECT": 0.0,
    "mult:harness": 0.5, "mult:docs": 0.2
  },
  "eligibility": {
    "min_credibility": 0.2,
    "min_valid_merged_prs": 0,
    "max_open_pr_threshold": 2
  },
  "maintainer_cut": 0.3
}
```

Rationale: sparkinfer's archetype (deterministic performance bot, pay for
verified frontier improvement only) plus two vanguarstew-style taxonomy
fallbacks so harness/docs labor isn't worthless. Lenient entry while the repo
is at 1% and needs miners; `max_open_pr_threshold` spam brake from day one;
0.3 cut (fund the eval GPU, signal reinvestment; raise with the share).

## Execution order

1. **Eval bot first** (the gating asset): re-run `python -m eval` on a pinned
   GPU per PR, post the deterministic `eval:*` label, append to a public
   `ledger.jsonl`, update the dashboard `data.json`. Adapt sparkinfer's
   MIT-licensed `eval/pr_eval_bot.py` + `bench/scripts/label.py`.
2. **Anti-gaming package**: per-eval random seeds (fixes the known
   `seed=0` overfit exploit in `eval/evaluator.py`), copycat diff-containment
   check, blocked-contributors flow, and a miner-guide CONTRIBUTING section
   stating exactly what scores and what scores zero.
3. **Winnable tracks** so labels can actually fire: `smooth` / `low-rank`
   matmul tracks now (DCT transform is the first admitted-strategy candidate),
   `attention-shaped` track next (bridge to the spectral roadmap, milestone M1).
4. **Run it in public for a few weeks** — real ledger entries, moving
   dashboard.
5. **File the weight_adjustment PR** to `entrius/gittensor` with that evidence.
   Pitch: same verification architecture as the 34% repo, pointed at a
   research-grade problem (sub-quadratic matmul/attention). The registry team
   explicitly invites repo recommendations, and merged retunes for four repos
   in the week of 2026-07-06 (#1572, #1578–#1586) show the process is active.

## Emission arithmetic (for the pitch)

Contributor pool = `emission_share × 0.90 × (1 − maintainer_cut)`.
At today's 1% with a 0.3 cut: 0.63% of subnet emissions to contributors.
sparkinfer's contributors split ≈ 15.4% — the gap is the case for the raise,
and verified frontier activity is the evidence that closes it.
