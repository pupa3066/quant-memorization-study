"""analysis.py — compute memorization/factuality metrics per precision from runs.jsonl.

Metrics (DESIGN.md §6-7):
- memorization GAP per precision = recon rate(memorized-candidate) - recon rate(control)
- factuality accuracy per precision, overall + by popularity
- paired McNemar (fp16 vs int8, fp16 vs int4) on QA correctness
- bootstrap CI for rate differences

Pure Python. Reads runs.jsonl; prints analysis.json. Meaningless on fabricated/empty data.
"""
from __future__ import annotations
import json, sys, math, random, os
from collections import defaultdict

def load(p): return [json.loads(l) for l in open(p) if l.strip()]

def rate(rows): return sum(1 for r in rows if r["correct"]) / len(rows) if rows else 0.0
def mean_score(rows): return sum(r.get("score", 0.0) for r in rows) / len(rows) if rows else 0.0

def mcnemar(pairs):
    b = c = 0
    for a, d in pairs:
        if a and not d: b += 1
        elif not a and d: c += 1
    n = b + c
    if n == 0: return b, c, 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    return b, c, round(chi2, 4), round(math.erfc(math.sqrt(chi2 / 2)), 4)

def boot_ci(diffs_src, n_boot=5000, alpha=0.05, seed=0):
    rng = random.Random(seed); m = len(diffs_src)
    if m == 0: return (0.0, 0.0, 0.0)
    base = sum(diffs_src) / m
    ds = []
    for _ in range(n_boot):
        s = [diffs_src[rng.randrange(m)] for _ in range(m)]
        ds.append(sum(s) / m)
    ds.sort()
    return (round(base, 4), round(ds[int(alpha/2*n_boot)], 4), round(ds[int((1-alpha/2)*n_boot)], 4))

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "runs.jsonl"
    rows = load(path)
    by = defaultdict(list)
    for r in rows: by[(r["probe"], r["precision"])].append(r)

    out = {"data_source": os.path.basename(path), "n_rows": len(rows), "per_precision": {}}

    precisions = sorted({r["precision"] for r in rows})
    for prec in precisions:
        mem = by[("mem", prec)]
        memc = [r for r in mem if r["item_id"].startswith("mem_")]
        ctrl = [r for r in mem if r["item_id"].startswith("ctrl_")]
        qa = by[("qa", prec)]
        qa_high = [r for r in qa if "high" in r["item_id"]]
        qa_low = [r for r in qa if "low" in r["item_id"]]
        out["per_precision"][prec] = {
            "mem_recon_frac_memorized": round(mean_score(memc), 4),
            "mem_recon_frac_control": round(mean_score(ctrl), 4),
            "mem_GAP": round(mean_score(memc) - mean_score(ctrl), 4),
            "mem_any_hit_memorized": round(rate(memc), 4),
            "qa_accuracy": round(rate(qa), 4),
            "qa_high_pop": round(rate(qa_high), 4),
            "qa_low_pop": round(rate(qa_low), 4),
            "n_mem": len(mem), "n_qa": len(qa),
        }

    # paired QA comparisons vs fp16
    def qa_map(prec):
        return {r["item_id"]: r["correct"] for r in by[("qa", prec)]}
    if "fp16" in precisions:
        base = qa_map("fp16")
        out["comparisons"] = {}
        for prec in precisions:
            if prec == "fp16": continue
            other = qa_map(prec)
            keys = sorted(set(base) & set(other))
            pairs = [(base[k], other[k]) for k in keys]
            b, c, chi2, p = mcnemar(pairs)
            diffs = [int(other[k]) - int(base[k]) for k in keys]
            out["comparisons"][f"qa_{prec}_vs_fp16"] = {
                "mcnemar": {"b": b, "c": c, "chi2": chi2, "p": p},
                "acc_diff_ci": dict(zip(("diff","lo","hi"), boot_ci(diffs))),
            }

    # ---- memorization comparison vs fp16 (paired reconstruction score per passage) ----
    def mem_score_map(prec):
        return {r["item_id"]: r.get("score", 0.0) for r in by[("mem", prec)]}
    if "fp16" in precisions and by[("mem", "fp16")]:
        base = mem_score_map("fp16")
        out.setdefault("comparisons", {})
        for prec in precisions:
            if prec == "fp16": continue
            other = mem_score_map(prec)
            keys = sorted(set(base) & set(other))
            # paired reconstruction-score diff (int_prec - fp16), memorized items only
            mem_keys = [k for k in keys if k.startswith("mem_")]
            diffs = [other[k] - base[k] for k in mem_keys]
            # GAP difference: (memGAP at prec) - (memGAP at fp16), via bootstrap over items
            out["comparisons"][f"mem_{prec}_vs_fp16"] = {
                "mem_recon_score_diff_ci": dict(zip(("diff","lo","hi"), boot_ci(diffs))),
                "n_memorized_items": len(mem_keys),
                "note": "diff = mean(recon_score[prec] - recon_score[fp16]) over memorized passages; <0 means precision reduced reconstruction",
            }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
