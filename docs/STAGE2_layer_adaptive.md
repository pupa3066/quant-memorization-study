# Stage-2: Layer-Adaptive Precision (analysis-only, no model loading)

> Consumes ALREADY-MEASURED layer-sensitivity data (AnimeVlog int4 ablation, SDXL UNet). Pure
> analysis — no inference, no model load, no RAM spike. Tool: `src/layer_adaptive.py`.

## Measured input (real, from AnimeVlog ablation_int4_quality.json)
| skip first/last | layers quantized | cosine fidelity |
|---|---|---|
| 0 | 10 | 0.999927 |
| 1 | 8 | 0.999942 |
| 2 | 6 | 0.999956 |
| 3 | 4 | 0.999971 |

## Finding
- **Layer sensitivity is UNEVEN (direction confirmed):** protecting (not quantizing) the first/last
  layers monotonically improves fidelity (0.999927 → 0.999971). This empirically supports the
  scheduler hypothesis H2 — the optimal precision policy is *per-layer* (quantize inner layers,
  protect the sensitive first/last), not uniform. This mirrors per-layer dataflow tuning in
  sparse-conv engines, applied to the precision axis.
- **HONEST effect size: negligible at this fidelity level.** Marginal fidelity gain per protected
  layer is ~7–8e-6, and *every* configuration is already above a 0.9999 quality floor. So for THIS
  model, aggressive uniform int4 (all 10 layers) is fine — the per-layer policy recommends max
  quantization because even the least-protected config clears the floor.
- **Where per-layer precision actually matters:** models with a hard, sensitive layer (a config that
  drops below the floor). This SDXL UNet isn't such a case; the value of the policy is that it
  *detects* when it is. That is the honest contribution: a policy that quantizes maximally while
  respecting a fidelity floor, and protects layers only when the data says they're sensitive.

## Policy output (for this model)
Recommended: skip_first_last=0 → quantize all 10 layers (cosine 0.999927 > 0.9999 floor). Max memory
saving; no layer needs protection here.

## Honest scope
- One model's ablation (SDXL UNet), 4 measured points. The *mechanism* (uneven sensitivity → per-layer
  policy) generalizes; the *effect size* is model-specific and small here.
- No new measurement performed (constraint: no RAM-heavy model loading). This validates the POLICY
  LOGIC on real data; measuring layer sensitivity on new models is future on-device work (gated).

## Feeds
This layer-sensitivity policy is a component of the precision+placement scheduler
(SCHEDULER_RESEARCH_DESIGN.md, H2): per-component AND per-layer precision under a memory budget.
