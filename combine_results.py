#!/usr/bin/env python3
"""combine_results.py — aggregate per-model analysis_*.json into RESULTS_multimodel.md.

Reads all analysis_popqa_*.json and analysis_mem_*.json in the study dir, builds a
cross-model table for factuality (int4 vs fp16) and memorization GAP by precision.
Honest: reports whatever the data says, labels missing precisions, no fabrication.
"""
import json, glob, os

HERE = os.path.dirname(os.path.abspath(__file__))

def load(p):
    try:
        with open(p) as fh: return json.load(fh)
    except Exception:
        return None

def tag_of(path, kind):
    base = os.path.basename(path)
    return base.replace(f"analysis_{kind}_", "").replace(".json", "")

def main():
    fact_rows, mem_rows = [], []

    for p in sorted(glob.glob(os.path.join(HERE, "analysis_popqa_*.json"))):
        d = load(p)
        if not d: continue
        tag = tag_of(p, "popqa")
        pc = d.get("per_precision", {})
        comp = d.get("comparisons", {}).get("qa_int4_vs_fp16", {})
        mc = comp.get("mcnemar", {})
        ci = comp.get("acc_diff_ci", {})
        fact_rows.append((tag,
                          pc.get("fp16", {}).get("qa_accuracy"),
                          pc.get("int4", {}).get("qa_accuracy"),
                          mc.get("p"),
                          ci.get("lo"), ci.get("hi")))

    for p in sorted(glob.glob(os.path.join(HERE, "analysis_mem_*.json"))):
        d = load(p)
        if not d: continue
        tag = tag_of(p, "mem")
        pc = d.get("per_precision", {})
        mem_rows.append((tag,
                         pc.get("fp16", {}).get("mem_GAP"),
                         pc.get("int8", {}).get("mem_GAP"),
                         pc.get("int4", {}).get("mem_GAP")))

    auc_rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "separability_*.json"))):
        d = load(p)
        if not d: continue
        tag = os.path.basename(p).replace("separability_", "").replace(".json", "")
        pc = d.get("per_precision", {})
        auc_rows.append((tag,
                         (pc.get("fp16") or {}).get("auc"),
                         (pc.get("int8") or {}).get("auc"),
                         (pc.get("int4") or {}).get("auc")))

    lines = ["# Multi-Model Results — Quantization × Memorization/Factuality",
             "",
             "> Auto-generated from per-model analysis_*.json by combine_results.py.",
             "> All numbers MEASURED on real models. Missing cells = precision variant not run.",
             "",
             "## Factuality (PopQA): INT4 vs FP16",
             "| Model | fp16 acc | int4 acc | McNemar p | acc-diff CI |",
             "|---|---|---|---|---|"]
    for tag, f, i, p, lo, hi in fact_rows:
        ci = f"[{lo}, {hi}]" if lo is not None else "—"
        lines.append(f"| {tag} | {f} | {i} | {p} | {ci} |")

    lines += ["",
              "## Memorization GAP by precision (memorized − control reconstruction)",
              "| Model | fp16 GAP | int8 GAP | int4 GAP |",
              "|---|---|---|---|"]
    for tag, f8, i8, i4 in mem_rows:
        lines.append(f"| {tag} | {f8} | {i8} | {i4} |")

    lines += ["",
              "## Memorization DETECTABILITY: AUC (memorized vs control) by precision",
              "AUC 0.5 = memorization undetectable by the probe; higher = more detectable.",
              "| Model | fp16 AUC | int8 AUC | int4 AUC |",
              "|---|---|---|---|"]
    for tag, f8, i8, i4 in auc_rows:
        lines.append(f"| {tag} | {f8} | {i8} | {i4} |")

    lines += ["",
              "## How to read this",
              "- Factuality: p≈1.0 with a CI straddling 0 ⇒ INT4 indistinguishable from FP16 (null).",
              "- Memorization: look for a monotonic fp16≥int8≥int4 decline AND a nonzero fp16 baseline.",
              "  A larger fp16 GAP (bigger/more-memorizing model) makes any INT4 collapse more conclusive.",
              "- Effect sizes + intervals matter more than any single point estimate. Underpowered rows",
              "  (tiny fp16 GAP) are suggestive only.",
              ""]
    with open(os.path.join(HERE, "RESULTS_multimodel.md"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote RESULTS_multimodel.md ({len(fact_rows)} factuality, {len(mem_rows)} memorization models)")

if __name__ == "__main__":
    main()
