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

## Systems finding: INT4 memory win is universal, but the LATENCY win depends on fused-kernel support
The most transferable systems observation from this work is *not* the memory saving (well known) — it
is that **whether INT4 is faster or slower is determined entirely by whether the runtime has a fused
low-bit matmul kernel**, measured across two independent implementations on the SAME Apple-Silicon host:

| implementation | INT4 memory | INT4 per-op / decode latency vs FP16 | why |
|---|---|---|---|
| MLX (fused int4 kernels) | ~72% smaller | **2.41× FASTER** decode | fused dequant-matmul; memory-bound win realized |
| PyTorch-MPS (no fused int4 kernel) | ~72% smaller (3.8×) | **6.3× SLOWER** per-op (0.75ms→4.75ms) | dequant overhead dominates; no fused Metal matmul |

**Takeaway (kernel/systems level):** the ~4× memory-footprint reduction from INT4 holds regardless of
implementation, but it only converts into a *latency* win when a fused low-bit kernel exists. Without
one, INT4 is a latency regression even though it's still a net win on a memory-bound device (it
eliminates weight offloading). This is a "needs a fused kernel / hardware support" gap — precisely the
layer a hardware-efficient-ML effort (custom Metal shader, PIM, or accelerator datapath) would close.
The PyTorch-MPS measurement is a per-op microbenchmark on one representative 2048×2048 Linear; the
absence of a fused MPS int4 kernel is implementation-specific, not fundamental.

*(The PyTorch-MPS numbers come from an independent on-device diffusion project; only the measured
figures are used here. Detail: `EFFICIENCY_animevlog.md`.)*

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
