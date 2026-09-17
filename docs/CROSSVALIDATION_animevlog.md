# AnimeVlog int4 Implementation × Quantization Study — Honest Evaluation

> Question asked: is AnimeVlog's int4 implementation the "ultimate solution to all quantization
> problems," can it test more cases, and what's best for AnimeVlog? Evaluated against real files.
> Sources: AnimeVlog/model/int4_mps.py, quantization_results.json, ablation_int4_quality.json,
> int4_publishability_assessment.json (all MEASURED, dated 2026-08-13).

## What the AnimeVlog int4 implementation actually is
- **Pure-PyTorch per-group asymmetric int4 weight-only quantization for MPS** (Apple Silicon),
  `Int4LinearMPS` with a `from_float` API. Packs 2×int4/byte + fp16 per-group scale/zero-point.
- **Measured:** SDXL UNet 2.57B params, fp16 4897MB → int4 ~1759MB (3.8× at group_size=128);
  cosine similarity 0.99778–0.99799 across group sizes 32–256; layer-sensitivity skip-first/last
  raises cosine to 0.99997; NF4 was WORSE than uniform here (nf4_rmse 0.00436 vs uniform 0.00201,
  −117%) because per-group calibration already handles the distribution.

## Is it the "ultimate solution to all quantization problems"? — NO.
Tested against evidence, including AnimeVlog's OWN assessment file:
- Its own `int4_publishability_assessment.json` says: "NOT publishable as standalone... engineering
  workaround, not algorithmic novelty... per-group asymmetric quantization is textbook... reproducible
  in 20 minutes." That is the correct, honest read.
- It solves ONE case well: diffusion UNet **weight-reconstruction fidelity** on MPS. It does NOT address:
  - activation quantization, KV-cache quantization, or mixed-precision search;
  - **model BEHAVIOR** (memorization, factuality) — it measures cosine of weights/outputs, a different
    dependent variable than the study;
  - non-MPS hardware, non-diffusion architectures beyond Linear layers (87% of this UNet).
- Verdict: a solid, reusable **engineering tool for one modality**, not a general solution. Claiming
  otherwise would contradict your own files and (per that assessment) "damage credibility."

## Can it be used to test MORE quantization cases? — YES, as a method.
Reusable and worth adopting into the study:
- The **group-size ablation** + **layer-sensitivity (skip first/last)** methodology is directly
  transferable: the study could report the same ablation axes for LLM quantization.
- `Int4LinearMPS.from_float` could quantize LLM Linear layers to probe how group size / skipped layers
  affect memorization & factuality (not just fidelity) — a genuinely new experiment the study lacks.
- ACTION (future): add a group-size axis to the study's harness, reusing this ablation design.

## What's best for AnimeVlog (per its own evidence)
- Keep int4 as a **supporting section (5.1)**, not the headline — the headline is Composable QLoRA +
  zero-interference. (Matches the assessment file.)
- Ship the int4 code as a package/util (MIT), not as a paper. Its value is engineering reuse.

## Cross-validation label (recorded in cross-links.md)
AnimeVlog int4 [MEASURED cosine >0.997 fidelity, vision] SUPPORTS quant-memorization-study
[MEASURED int4 factuality null, text] on "int4 preserves intended task output across modalities";
NEUTRAL on memorization (fidelity ≠ verbatim recall). Cross-modality external validity for the study.
