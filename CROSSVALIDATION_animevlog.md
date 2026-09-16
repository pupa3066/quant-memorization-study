# Cross-Validation: Quant-Memorization Study × AnimeVlog

> Ran the bootstrap cross-validation step. MEASURED data only, no hypothetical bridges.
> Both projects use INT4 as an independent variable but measure DIFFERENT dependent variables.

## The two evidence sets (both MEASURED)
- **AnimeVlog (int4 on SDXL/diffusion):** output FIDELITY. int4 gives 64% memory saving
  (UNet 4897MB→1768MB) at cosine >0.997 (0.99778–0.99799). int4 preserves image-latent fidelity.
  int4 alone judged "NOT publishable" as a standalone contribution.
- **Quant-memorization study (int4 on LLMs, 6 models):** BEHAVIOR. int4 ≈ fp16 on factual accuracy
  (replicated null, McNemar p=1.0/1.0/0.38); int4's effect on memorization is inconsistent and
  dominated by model scale.

## Relationship labels
- **SUPPORTS (capability-preservation axis):** Both independently show int4 preserves the model's
  primary useful output — AnimeVlog on image fidelity (cosine >0.997), the study on factual accuracy
  (null). Two different modalities (diffusion vision vs. LLM text), same conclusion: **int4 is a safe
  compression for the intended task output.** This is real mutual support, not a stretch.
- **NEUTRAL (memorization):** AnimeVlog measures no memorization signal (it measures latent fidelity);
  the study's memorization result is about verbatim training-data recall. High cosine fidelity does
  NOT imply preserved memorization, and the study's mixed memorization result does NOT contradict
  AnimeVlog's fidelity numbers. Different dependent variables — correctly labeled NEUTRAL.
- **REFINES (AnimeVlog's "int4 not publishable alone"):** AnimeVlog concluded int4-alone isn't a
  publishable contribution (it's just a known compression). The study REFINES that: int4-alone is
  indeed unremarkable for *capability*, but int4's effect on *memorization/behavior* is a genuinely
  under-studied, measurable question — the publishable angle isn't the compression, it's what
  behavior it does/doesn't change. Consistent with AnimeVlog's own call, and sharpens why.

## Net
No contradiction. The two projects agree on the defensible claim ("int4 preserves task capability")
across vision and text, and are properly orthogonal on memorization. AnimeVlog's "int4 not
publishable alone" stance stands and is refined, not overturned. No stale claim to fix.

## Edge recorded in cross-links.md
quant-memorization-study [MEASURED int4 factuality null, 6 models] SUPPORTS AnimeVlog [MEASURED int4
cosine >0.997 fidelity] on "int4 preserves intended output"; NEUTRAL on memorization (different DV).
