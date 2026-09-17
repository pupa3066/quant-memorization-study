# Efficiency × Behavior Tradeoff — AnimeVlog (SDXL diffusion)

> Stage-1 efficiency analysis applied to a DIFFUSION model (SDXL UNet), not an LLM. Metrics adapted:
> per-op latency + memory footprint + cosine-fidelity (behavior axis for diffusion) instead of
> tokens/sec + QA accuracy. All numbers MEASURED on Apple M1 8GB; the int4-vs-fp16 benchmark was
> RE-RUN fresh and reproduces AnimeVlog's logged data. Sources: AnimeVlog int4_mps.py benchmark,
> quantization_experiment/int4_mps_results.json, profiling/full_pipeline_profile.json.

## Measured (SDXL UNet, per-op 2048×2048 Linear, MPS)
| precision | memory | mem reduction | per-op latency | speed ratio | cosine fidelity |
|---|---|---|---|---|---|
| fp16 | 8.00 MB | — | 0.75 ms | 1.00× | baseline |
| int4 | 2.12 MB | 3.8× (72%) | 4.75 ms | **0.16× (6.3× SLOWER)** | 0.9978 |

Full SDXL UNet: fp16 4897 MB → int4 1768 MB (**~64% saving**) → fits 8GB with VAE+TE, no offloading.

## The key cross-implementation finding (why testing AnimeVlog mattered)
Compare to the LLM Stage-1 result (MLX Qwen2.5-0.5B): there INT4 was **2.4× FASTER**. Here INT4 is
**6.3× SLOWER per-op**. Same memory win (~3.8×), OPPOSITE latency direction. Reason (measured +
documented in AnimeVlog): **MLX has fused int4 kernels; PyTorch-MPS does not**, so SDXL int4 pays
dequantization overhead with no fused Metal matmul.

**Systems takeaway (directly in the SPIN/efficiency idiom):** the int4 memory win is universal, but
whether it's a *latency* win depends entirely on **kernel support**. On a memory-bound 8GB device the
net effect is still positive (int4 eliminates offloading, avoiding the ~17-min offloaded generation);
on a compute-bound setup without fused kernels, int4 can be a latency *regression*. This is exactly a
"needs a fused kernel / hardware support" story — the gap a hardware-efficient-ML group addresses.

## Behavior axis (diffusion analogue of memorization/factuality)
- Fidelity preserved: cosine 0.9978 (>0.997) — int4 keeps the intended image output, mirroring the
  LLM factuality null. So on BOTH modalities, int4 preserves the intended task output.
- (Memorization has no direct diffusion analogue here; fidelity ≠ verbatim recall — see cross-val.)

## Honest scope
- Per-op microbenchmark (one representative Linear) + logged full-UNet footprint; not a full
  end-to-end int4 generation latency (the pipeline profile is fp16-offloaded, 470s @ M1 8GB).
- The "no fused MPS int4 kernel" penalty is implementation-specific (PyTorch-MPS), not fundamental —
  a fused Metal shader would remove it (AnimeVlog lists this as future work).

## Why this strengthens the efficiency research
It turns the single-model LLM tradeoff into a **cross-modality, cross-implementation** result:
- int4 memory win holds across LLM (text) and SDXL (image): ~72% / ~64%.
- int4 preserves intended output across both (factuality null / cosine >0.997).
- int4 latency direction FLIPS with kernel support (MLX fused → faster; PyTorch-MPS → slower).
That third point is a genuine systems contribution and a natural bridge to hardware-efficient-ML work.
