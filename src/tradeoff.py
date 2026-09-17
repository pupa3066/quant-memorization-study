#!/usr/bin/env python3
"""tradeoff.py — join SYSTEMS efficiency (efficiency_*.json) with BEHAVIOR (analysis_*.json)
into one behavior-vs-efficiency table per precision, for a single model tag.

This is the bridge result: it shows, per precision, what you PAY (memory/latency) vs what you
LOSE/KEEP in behavior (factuality, memorization). The point neither pure-systems nor pure-NLP
work usually makes.

Usage: tradeoff.py <tag>   (e.g. qwen05 -> reads efficiency_qwen05.json, analysis_popqa_qwen05.json,
                            analysis_mem_qwen05.json)
"""
import json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")

def load(p):
    try: return json.load(open(os.path.join(RESULTS, p)))
    except Exception: return None

def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "qwen05"
    eff = load(f"efficiency_{tag}.json")
    qa = load(f"analysis_popqa_{tag}.json")
    mem = load(f"analysis_mem_{tag}.json")
    if not eff:
        print(f"no efficiency_{tag}.json", file=sys.stderr); sys.exit(1)

    qa_pc = (qa or {}).get("per_precision", {})
    mem_pc = (mem or {}).get("per_precision", {})

    rows = []
    for r in eff["per_precision"]:
        p = r["precision"]
        rows.append({
            "precision": p,
            "weight_MB": r.get("weight_mb"),
            "mem_reduction_vs_fp16": r.get("weight_reduction_vs_fp16"),
            "decode_tok_s": r.get("decode_tok_s"),
            "decode_speedup_vs_fp16": r.get("decode_speedup_vs_fp16"),
            "qa_accuracy": qa_pc.get(p, {}).get("qa_accuracy"),
            "mem_GAP": mem_pc.get(p, {}).get("mem_GAP"),
        })

    print(f"# Behavior-vs-Efficiency tradeoff — {tag}\n")
    hdr = f"| precision | weight_MB | mem_reduction | tok/s | speedup | QA_acc | mem_GAP |"
    print(hdr); print("|" + "---|"*7)
    for r in rows:
        print(f"| {r['precision']} | {r['weight_MB']} | {r['mem_reduction_vs_fp16']} | "
              f"{r['decode_tok_s']} | {r['decode_speedup_vs_fp16']} | {r['qa_accuracy']} | {r['mem_GAP']} |")
    json.dump({"tag": tag, "rows": rows}, open(os.path.join(RESULTS, f"tradeoff_{tag}.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
