# Cross-Hardware Efficiency Results

> Auto-generated from results/hw_*.json. Each row = one hardware/backend, MEASURED with
> a comparable harness. Blank = not measured. Contributors credited per `measured_by`.

## Provenance & attribution
| hardware | backend | fused int4? | measured_by | harness |
|---|---|---|---|---|
| Apple M1 8GB | MLX (fused int4 kernels) | True | Purnima Pathak | efficiency.py (this repo) |

## INT4 vs FP16 per hardware
| hardware | fused int4? | int4 mem reduction | int4 decode speedup vs fp16 |
|---|---|---|---|
| Apple M1 8GB | True | 0.719 | 2.41× |

## Read honestly
- Memory reduction should be ~constant across hardware (precision, not device, sets it).
- Decode speedup should track `fused int4?`: >1× where fused kernel exists, <1× where not.
- State the cross-hardware claim ONLY if the measured pattern actually holds. Report
  contradictions if they appear — that's a finding too.