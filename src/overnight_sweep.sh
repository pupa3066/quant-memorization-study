#!/usr/bin/env bash
# overnight_sweep.sh — RAM/disk-safe multi-model quantization x memorization/factuality sweep.
#
# Safety design (per Pupa's constraints — "without hampering my computer or RAM"):
#  - SEQUENTIAL only: one model loaded at a time, in a fresh python process that EXITS
#    (frees all RAM) before the next model. Never parallel.
#  - PRE-FLIGHT per model: check free disk >= MIN_DISK_GB and skip if not.
#  - DELETE-AFTER-USE: remove each model from the HF cache once measured (approved).
#  - Small models only (fit 8GB). If a load fails (OOM/other), log + SKIP, never crash the loop.
#  - Results-only auto-commit to the PRIVATE repo; NEVER commits weights/venv/logs.
#  - All verbose output to a log file, not the terminal.
set -u
cd "$(dirname "$0")"
STUDY=~/Projects/quant-memorization-study
PY="$STUDY/.venv/bin/python"
LOG="$STUDY/overnight.log"
MIN_DISK_GB=8            # skip a model if free disk below this
POPQA_PER_BIN=50
MEM_CORPUS="$STUDY/data/mem_corpus.json"
RES="$STUDY/results"

# Model ladder: distinct small instruct models, 4-bit where possible (fit 8GB).
# fp16/int8/int4 triples only where a same-family variant exists; else single precision.
MODELS=(
  "qwen05:mlx-community/Qwen2.5-0.5B-Instruct-bf16:mlx-community/Qwen2.5-0.5B-Instruct-8bit:mlx-community/Qwen2.5-0.5B-Instruct-4bit"
  "qwen15:mlx-community/Qwen2.5-1.5B-Instruct-bf16::mlx-community/Qwen2.5-1.5B-Instruct-4bit"
  "llama1b:mlx-community/Llama-3.2-1B-Instruct-bf16::mlx-community/Llama-3.2-1B-Instruct-4bit"
  "qwen3b:::mlx-community/Qwen2.5-3B-Instruct-4bit"
  "phi35:::mlx-community/Phi-3.5-mini-instruct-4bit"
  "gemma2b:::mlx-community/gemma-2-2b-it-4bit"
)
# NOTE: all paths verified to resolve on HF (HTTP 200) on 2026-09-15. 3B/mini are 4-bit only
# (~1.8-2.3GB weights) to stay within 8GB RAM. Larger fp16 models intentionally excluded (won't fit).

free_disk_gb() { df -g "$HOME" | awk 'NR==2{print $4}'; }

log() { echo "[$(date '+%H:%M:%S')] $*" >>"$LOG"; }

log "=== overnight sweep start ==="
log "free disk: $(free_disk_gb) GB"

for entry in "${MODELS[@]}"; do
  IFS=':' read -r tag fp16 int8 int4 <<<"$entry"
  # rejoin the model paths (they contain no colon in the org/name after mlx-community/, but the
  # scheme uses ':' as delimiter; mlx paths have none, so this is safe)
  free=$(free_disk_gb)
  if [ "$free" -lt "$MIN_DISK_GB" ]; then
    log "SKIP $tag — free disk ${free}GB < ${MIN_DISK_GB}GB"
    continue
  fi
  log "--- model $tag (fp16='$fp16' int8='$int8' int4='$int4') ---"

  args=()
  [ -n "$fp16" ] && args+=(--fp16 "$fp16")
  [ -n "$int8" ] && args+=(--int8 "$int8")
  [ -n "$int4" ] && args+=(--int4 "$int4")

  # PopQA factuality run
  if "$PY" "$STUDY/src/harness.py" --run "${args[@]}" --popqa "$POPQA_PER_BIN" \
        --out "$RES/runs_popqa_${tag}.jsonl" >>"$LOG" 2>&1; then
    "$PY" "$STUDY/src/analysis.py" "$RES/runs_popqa_${tag}.jsonl" > "$RES/analysis_popqa_${tag}.json" 2>>"$LOG"
    log "  popqa OK -> analysis_popqa_${tag}.json"
  else
    log "  popqa FAILED for $tag (skipping, loop continues)"
  fi

  # Memorization run
  if "$PY" "$STUDY/src/harness.py" --run "${args[@]}" --mem-corpus "$MEM_CORPUS" \
        --out "$RES/runs_mem_${tag}.jsonl" >>"$LOG" 2>&1; then
    "$PY" "$STUDY/src/analysis.py" "$RES/runs_mem_${tag}.jsonl" > "$RES/analysis_mem_${tag}.json" 2>>"$LOG"
    "$PY" "$STUDY/src/separability.py" "$RES/runs_mem_${tag}.jsonl" > "$RES/separability_${tag}.json" 2>>"$LOG"
    log "  mem OK -> analysis_mem_${tag}.json + separability_${tag}.json"
  else
    log "  mem FAILED for $tag (skipping, loop continues)"
  fi

  # DELETE-AFTER-USE: remove this model's weights from the HF cache (approved).
  for mp in "$fp16" "$int8" "$int4"; do
    [ -z "$mp" ] && continue
    # HF cache dir name convention: models--org--name
    cache_name="models--$(echo "$mp" | sed 's#/#--#g')"
    d="$HOME/.cache/huggingface/hub/$cache_name"
    if [ -d "$d" ]; then rm -rf "$d" && log "  deleted cache $cache_name"; fi
  done
  log "  free disk after cleanup: $(free_disk_gb) GB"

  # Results-only auto-commit (NEVER weights/venv/logs — .gitignore enforces).
  ( cd "$STUDY" && git add results/runs_*_"${tag}".jsonl results/analysis_*_"${tag}".json results/separability_"${tag}".json 2>/dev/null \
      && git commit -q -m "overnight: add $tag runs (factuality+memorization)" 2>>"$LOG" \
      && git push -q 2>>"$LOG" && echo "committed+pushed $tag" >>"$LOG" ) || log "  git step skipped/failed for $tag"
done

# Final combined analysis writeup
"$PY" "$STUDY/src/combine_results.py" >>"$LOG" 2>&1 || log "combine step skipped"
( cd "$STUDY" && git add RESULTS_multimodel.md 2>/dev/null && git commit -q -m "overnight: combined multi-model results" 2>>"$LOG" && git push -q 2>>"$LOG" ) || true
log "=== overnight sweep done ==="
