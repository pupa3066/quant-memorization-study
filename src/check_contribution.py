#!/usr/bin/env python3
"""check_contribution.py — validate cross-hardware data contributions before a PR is merged.

Enforces the rules in docs/CROSS_HARDWARE_SCHEMA.md so contributed data is COMPARABLE and
attributed — protecting the study's integrity. Runs in CI or locally:

    python3 src/check_contribution.py                 # check all results/hw_*.json
    python3 src/check_contribution.py results/hw_x.json  # check specific files

Exit 0 = all pass (safe to merge). Exit 1 = at least one FAIL (block merge).
Checks are HONEST gates, not fabrication: it verifies structure/provenance/attribution and flags
values that look physically impossible or incomparable — it does NOT invent or "fix" data.
"""
import json, sys, glob, os, math

REQUIRED_TOP = ["hardware", "backend", "fused_int4_kernel", "measured_by", "harness_version",
                "model", "per_precision"]
REQUIRED_PP  = ["precision", "weight_mb"]          # minimally required per precision row
KNOWN_PREC   = {"fp16", "bf16", "int8", "int4"}

def check_file(path):
    errs, warns = [], []
    try:
        d = json.load(open(path))
    except Exception as e:
        return [f"not valid JSON: {e}"], []

    # 1. schema: required top-level fields
    for k in REQUIRED_TOP:
        if k not in d or d[k] in (None, "", "?"):
            errs.append(f"missing/blank required field: '{k}'")

    # 2. attribution (integrity): measured_by must be a real name, not placeholder
    mb = str(d.get("measured_by", "")).strip()
    if mb.lower() in ("", "?", "todo", "anonymous", "unknown"):
        errs.append("attribution: 'measured_by' must name the contributor (integrity requirement)")

    # 3. provenance: harness_version present (so data is traceable/comparable)
    if not str(d.get("harness_version", "")).strip():
        errs.append("provenance: 'harness_version' required (comparability)")

    # 4. fused_int4_kernel must be an explicit bool (it's the key explanatory variable)
    if not isinstance(d.get("fused_int4_kernel"), bool):
        errs.append("'fused_int4_kernel' must be true/false (explicit) — it explains latency direction")

    # 5. per_precision structure + sanity
    pp = d.get("per_precision", [])
    if not isinstance(pp, list) or not pp:
        errs.append("'per_precision' must be a non-empty list")
    else:
        precs = []
        fp16_mb = None
        for i, r in enumerate(pp):
            for k in REQUIRED_PP:
                if k not in r or r[k] in (None, ""):
                    errs.append(f"per_precision[{i}] missing '{k}'")
            p = r.get("precision")
            precs.append(p)
            if p not in KNOWN_PREC:
                warns.append(f"per_precision[{i}] unknown precision '{p}'")
            wm = r.get("weight_mb")
            if isinstance(wm, (int, float)):
                if wm <= 0: errs.append(f"per_precision[{i}] weight_mb must be > 0")
                if p in ("fp16", "bf16"): fp16_mb = wm
            # latency sanity: non-negative, not absurd
            for f in ("load_s", "prefill_ms", "decode_tok_s", "peak_mem_mb"):
                v = r.get(f)
                if isinstance(v, (int, float)) and v < 0:
                    errs.append(f"per_precision[{i}] '{f}' is negative ({v})")
        # 6. comparability sanity: int4 should be SMALLER than fp16 (physics of quantization)
        i4 = next((r for r in pp if r.get("precision") == "int4"), None)
        if fp16_mb and i4 and isinstance(i4.get("weight_mb"), (int, float)):
            if i4["weight_mb"] >= fp16_mb:
                errs.append(f"int4 weight_mb ({i4['weight_mb']}) >= fp16 ({fp16_mb}) — physically wrong; "
                            "check measurement (int4 must be smaller)")
        # 7. need a baseline to compare against
        if fp16_mb is None:
            warns.append("no fp16/bf16 baseline row — cross-hardware ratios can't be computed for this file")

    return errs, warns

def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "hw_*.json")))
    if not paths:
        print("no hw_*.json contribution files found (nothing to check)"); return 0
    total_fail = 0
    for p in paths:
        errs, warns = check_file(p)
        name = os.path.basename(p)
        if errs:
            total_fail += 1
            print(f"❌ FAIL  {name}")
            for e in errs: print(f"     - {e}")
        else:
            print(f"✅ PASS  {name}")
        for w in warns: print(f"     ⚠  {w}")

    # Integrity check for ANY results contribution: CITATION.cff must credit >1 author/contributor
    # when external data is added (a contributor's data must not be presented as the maintainer's own).
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cff = os.path.join(root, "CITATION.cff")
    if os.path.exists(cff):
        txt = open(cff).read()
        n_people = txt.count("given-names:")
        print(f"\n[attribution] CITATION.cff lists {n_people} person(s).")
        print("     ⚠  If this PR adds data measured by someone else, they MUST be added to "
              "CITATION.cff authors/contributors before merge (integrity). Verify manually.")

    if total_fail:
        print(f"\n{total_fail} file(s) FAILED — do NOT merge until fixed. See docs/CROSS_HARDWARE_SCHEMA.md.")
        return 1
    print("\nHardware-schema files pass. NOTE: non-hw_*.json result files are not schema-validated here; "
          "reviewer must confirm provenance (same harness?) + attribution before merge.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
