# Quantization × Memorization & Factuality

Does reducing a model's numeric precision (FP16 → INT4) change **what it memorizes** and **how
reliably it answers factual questions**? A small, honest, reproducible study bridging on-device
quantization with data-centric interpretability (Ravichander et al., arXiv:2503.12072).

> **Status: PILOT.** The apparatus runs end-to-end on real models and shows a directionally
> consistent effect, but N is tiny and results are NOT statistically significant. See DEVLOG.md §4.

## Files
- `DESIGN.md` — pre-registered design: RQs, hypotheses (H1–H3), metrics, statistics, threats.
- `harness.py` — measurement harness. Memorization probe (token-ID-level high-surprisal
  reconstruction) + factuality probe (popularity-stratified closed-book QA). MLX backend.
- `analysis.py` — per-precision metrics: memorization GAP, factuality accuracy by popularity,
  paired McNemar, bootstrap CIs.
- `DEVLOG.md` — deep engineering log: every bug, its root cause, the fix, and design rationale.
- `runs.jsonl` / `analysis.json` — the current pilot data + computed metrics (real, labeled pilot).

## Reproduce (Apple Silicon)
```sh
python3 -m venv .venv
./.venv/bin/python -m pip install numpy mlx mlx-lm
# dry plan (no model, no cost):
./.venv/bin/python harness.py --dry
# real run (downloads two precisions of the same small model):
./.venv/bin/python harness.py --run \
  --fp16 mlx-community/Qwen2.5-0.5B-Instruct-bf16 \
  --int4 mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --out runs.jsonl
./.venv/bin/python analysis.py runs.jsonl
```

## Current results
**Multi-model (6 models: Qwen2.5-0.5B/1.5B/3B, Llama-3.2-1B, Phi-3.5-mini, Gemma-2-2b) — see
`RESULTS_multimodel.md`:**
- **Factuality: INT4 ≈ FP16, a replicated null.** All 3 models with paired fp16/int4 data show no
  significant change in factual accuracy (McNemar p=1.0, 1.0, 0.38).
- **Memorization: dominated by model scale, not precision.** The single-0.5B-model "int4 erases
  memorization" story did NOT replicate across models (2/3 decline with int4, llama1b increases
  0.065→0.074; larger int4 models memorize more than small fp16 models). Honest, mixed result.
- **Methodology:** an N=8 pilot false positive (factuality) and a single-model memorization artifact
  were both corrected by scaling to 100 facts and 6 models. Measure, don't assert.

Reproduce: `overnight_sweep.sh` (RAM/disk-safe multi-model sweep) → `combine_results.py`. See DEVLOG.md.

## Honesty policy
No fabricated numbers. No backend → prints plan and exits. Empty data → zeros, not invented results.
