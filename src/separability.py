#!/usr/bin/env python3
"""separability.py — T2 metric: how well does reconstruction score SEPARATE memorized from control
text, at each precision? This is a membership/contamination-style detection metric.

Signal: AUC (area under ROC) of using per-passage reconstruction score to classify
memorized (positive) vs control (negative). AUC=0.5 -> no separability (memorization undetectable);
AUC->1.0 -> strong separability (memorization clearly detectable). If AUC declines with lower
precision, quantization is erasing the detectability of memorized data — a real interpretability
finding aligned to data-centric interpretability (Ravichander et al., 2025, arXiv:2503.12072).

Pure Python (no sklearn). Reads a runs_mem_*.jsonl; prints AUC per precision.
Usage: separability.py runs_mem_<tag>.jsonl
"""
import json, sys, os
from collections import defaultdict

def load(p): return [json.loads(l) for l in open(p) if l.strip()]

def auc(pos_scores, neg_scores):
    """Mann-Whitney U -> AUC. Ties count 0.5. Returns None if a class is empty."""
    if not pos_scores or not neg_scores:
        return None
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n: wins += 1
            elif p == n: wins += 0.5
    return round(wins / (len(pos_scores) * len(neg_scores)), 4)

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "runs_mem.jsonl"
    rows = [r for r in load(path) if r.get("probe") == "mem"]
    by_prec = defaultdict(lambda: {"pos": [], "neg": []})
    for r in rows:
        prec = r["precision"]
        s = r.get("score", 0.0)
        if r["item_id"].startswith("mem_"):
            by_prec[prec]["pos"].append(s)
        elif r["item_id"].startswith("ctrl_"):
            by_prec[prec]["neg"].append(s)

    out = {"data_source": os.path.basename(path), "metric": "AUC(memorized vs control reconstruction score)",
           "interpretation": "0.5 = memorization undetectable; higher = more detectable", "per_precision": {}}
    for prec in sorted(by_prec):
        d = by_prec[prec]
        out["per_precision"][prec] = {
            "auc": auc(d["pos"], d["neg"]),
            "n_memorized": len(d["pos"]), "n_control": len(d["neg"]),
            "mean_pos": round(sum(d["pos"])/len(d["pos"]), 4) if d["pos"] else None,
            "mean_neg": round(sum(d["neg"])/len(d["neg"]), 4) if d["neg"] else None,
        }
    # trend note
    aucs = [(p, v["auc"]) for p, v in out["per_precision"].items() if v["auc"] is not None]
    if len(aucs) >= 2:
        order = {"fp16": 0, "int8": 1, "int4": 2}
        aucs.sort(key=lambda x: order.get(x[0], 9))
        vals = [a for _, a in aucs]
        out["trend"] = ("declines with lower precision (memorization detectability erodes)"
                        if all(vals[i] >= vals[i+1] for i in range(len(vals)-1))
                        else "non-monotonic / inconclusive")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
