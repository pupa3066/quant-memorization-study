# Efficiency × Behavior Tradeoff (measured)

> Systems-side measurement paired with the behavior results, in the idiom of efficiency/systems-for-ML
> work (memory footprint, latency, throughput). All numbers MEASURED on MLX Qwen2.5-0.5B-Instruct,
> same model at three precisions. Reproduce: `efficiency.py` → `tradeoff.py`.

## Result (qwen05, fp16 vs int8 vs int4)
| precision | weight MB | mem reduction vs fp16 | decode tok/s | speedup | QA accuracy | memorization GAP |
|---|---|---|---|---|---|---|
| fp16 | 942 | — | 47.7 | 1.00× | 0.29 | 0.025 |
| int8 | 501 | 47% | 58.2 | 1.22× | 0.31 | 0.017 |
| int4 | 265 | **72%** | 114.9 | **2.41×** | 0.28 | 0.000 |

Prefill latency (single forward over a fixed prompt): fp16 202ms → int8 114ms → int4 **37.5ms**.

## The point (bridge between efficiency and behavior)
- **INT4 buys a lot and costs little on capability:** 72% smaller weights + 2.41× decode throughput +
  ~5.4× faster prefill, while **factual accuracy is statistically unchanged** (0.29→0.28, replicated
  null in the behavior study). This is the memory-bound win that systems-efficiency work targets.
- **What DOES change is behavior you can only see by measuring it:** the memorization reconstruction
  GAP fades (0.025→0.000 at this scale). Efficiency metrics alone would never surface this.
- **Thesis:** efficiency interventions should be evaluated on BOTH axes — systems cost (memory/latency)
  AND model behavior (memorization/factuality). Optimizing one blind to the other is incomplete.

## Honest scope
- Single small model (0.5B). Speedups/footprint are real and reproducible here; larger models will
  differ in absolute numbers (bigger memory-bound wins, stronger memorization baselines).
- `peak_rss_mb` is process-level (host), a coarse footprint proxy; `weight_MB` (on-disk quantized
  size) is the cleaner memory-footprint signal.
- Latency measured on Apple Silicon MLX, batch 1. Energy NOT measured (needs instrumented hardware /
  edge platform under a power budget — the natural next step on real systems).

## Next (extends efficiency-systems work)
1. Run the same efficiency×behavior sweep on **edge hardware under a power budget** to add ENERGY —
   the axis a laptop can't measure.
2. **Layer-adaptive precision:** quantize layers by structural sensitivity (analogous to per-layer
   dataflow tuning in sparse-conv engines) and measure memory saved vs behavior preserved.
3. Larger models where the memory-bound win and memorization baseline are both large.
