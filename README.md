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

## Current pilot signal (N=8, p=0.48 — pilot only)
- Memorization GAP: fp16 +0.10 → int4 0.00 (signal disappears under INT4).
- Factuality: high-popularity facts unaffected (1.0→1.0); long-tail degrades (0.67→0.33).
- Direction matches hypotheses; significance requires scaling N (see DEVLOG §5).

## Honesty policy
No fabricated numbers. No backend → prints plan and exits. Empty data → zeros, not invented results.
