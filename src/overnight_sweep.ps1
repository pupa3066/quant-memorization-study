<#
overnight_sweep.ps1 — Windows + NVIDIA (CUDA) multi-model quantization x memorization/factuality sweep.
LOCAL ONLY: models load from local directories (no network). Mirrors overnight_sweep.sh.

Safety design (same constraints as the bash sweep):
 - SEQUENTIAL only: one model per fresh python process that exits (frees VRAM/RAM) before the next.
 - PRE-FLIGHT per model: skip if free disk < MinDiskGB.
 - Small models sized to your VRAM. If a load fails (OOM/other), log + SKIP, never crash the loop.
 - Results-only auto-commit; NEVER commits weights/venv/logs (.gitignore enforces).
 - All verbose output to a log file.

Usage:
  .\src\overnight_sweep.ps1
  .\src\overnight_sweep.ps1 -Study C:\path\to\quant-memorization-study -ModelsRoot C:\models
Set $env:HF_TOKEN before running if any local model came from a gated repo.
#>
param(
  [string]$Study     = (Split-Path -Parent $PSScriptRoot),
  [string]$ModelsRoot = "C:\models",
  [int]   $MinDiskGB  = 20,
  [int]   $PopqaPerBin = 50
)
$ErrorActionPreference = "Continue"
$env:HF_HUB_OFFLINE   = "1"
$env:TRANSFORMERS_OFFLINE = "1"

$PY  = Join-Path $Study ".venv\Scripts\python.exe"
if (-not (Test-Path $PY)) { $PY = "python" }   # fall back to PATH python
$LOG = Join-Path $Study "overnight.log"
$RES = Join-Path $Study "results"
$MEM = Join-Path $Study "data\mem_corpus.json"
$POPQA_LOCAL = Join-Path $Study "data\popqa_local.jsonl"   # provide your local PopQA dump here

function Log([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format HH:mm:ss), $msg
  Add-Content -Path $LOG -Value $line
}
function FreeDiskGB([string]$path) {
  $drive = (Get-Item $path).PSDrive.Name
  [math]::Floor((Get-PSDrive $drive).Free / 1GB)
}

# Model ladder: local directory names under $ModelsRoot. Tag:fp16dir:int8dir:int4dir
# Leave a slot empty to skip that precision. For CUDA/bitsandbytes, int8/int4 quantize the
# SAME fp16 weights at load time — so point int8/int4 at the same fp16 folder.
$MODELS = @(
  @{ tag="qwen05";  fp16="Qwen2.5-0.5B-Instruct"; int8="Qwen2.5-0.5B-Instruct"; int4="Qwen2.5-0.5B-Instruct" },
  @{ tag="qwen15";  fp16="Qwen2.5-1.5B-Instruct"; int8="";                       int4="Qwen2.5-1.5B-Instruct" },
  @{ tag="llama1b"; fp16="Llama-3.2-1B-Instruct"; int8="";                       int4="Llama-3.2-1B-Instruct" },
  @{ tag="qwen3b";  fp16="";                       int8="";                       int4="Qwen2.5-3B-Instruct" },
  @{ tag="phi35";   fp16="";                       int8="";                       int4="Phi-3.5-mini-instruct" },
  @{ tag="gemma2b"; fp16="";                       int8="";                       int4="gemma-2-2b-it" }
)

Log "=== overnight sweep start (Windows/CUDA, local) ==="
Log ("free disk: {0} GB" -f (FreeDiskGB $Study))

foreach ($m in $MODELS) {
  $tag = $m.tag
  $free = FreeDiskGB $Study
  if ($free -lt $MinDiskGB) { Log "SKIP $tag — free disk ${free}GB < ${MinDiskGB}GB"; continue }

  $args = @()
  if ($m.fp16) { $args += @("--fp16", (Join-Path $ModelsRoot $m.fp16)) }
  if ($m.int8) { $args += @("--int8", (Join-Path $ModelsRoot $m.int8)) }
  if ($m.int4) { $args += @("--int4", (Join-Path $ModelsRoot $m.int4)) }
  Log "--- model $tag (args: $($args -join ' ')) ---"

  # Factuality run (local PopQA if present, else built-in QA set)
  $popqaArgs = @()
  if (Test-Path $POPQA_LOCAL) { $popqaArgs = @("--popqa", $PopqaPerBin, "--popqa-local", $POPQA_LOCAL) }
  & $PY (Join-Path $Study "src\harness.py") --run --backend hf @args @popqaArgs `
      --out (Join-Path $RES "runs_popqa_$tag.jsonl") *>> $LOG
  if ($LASTEXITCODE -eq 0) {
    & $PY (Join-Path $Study "src\analysis.py") (Join-Path $RES "runs_popqa_$tag.jsonl") `
        > (Join-Path $RES "analysis_popqa_$tag.json") 2>> $LOG
    Log "  popqa OK -> analysis_popqa_$tag.json"
  } else { Log "  popqa FAILED for $tag (loop continues)" }

  # Memorization run
  & $PY (Join-Path $Study "src\harness.py") --run --backend hf @args `
      --mem-corpus $MEM --out (Join-Path $RES "runs_mem_$tag.jsonl") *>> $LOG
  if ($LASTEXITCODE -eq 0) {
    & $PY (Join-Path $Study "src\analysis.py")     (Join-Path $RES "runs_mem_$tag.jsonl") > (Join-Path $RES "analysis_mem_$tag.json")   2>> $LOG
    & $PY (Join-Path $Study "src\separability.py") (Join-Path $RES "runs_mem_$tag.jsonl") > (Join-Path $RES "separability_$tag.json")   2>> $LOG
    Log "  mem OK -> analysis_mem_$tag.json + separability_$tag.json"
  } else { Log "  mem FAILED for $tag (loop continues)" }

  # Results-only auto-commit (NEVER weights/venv/logs — .gitignore enforces).
  Push-Location $Study
  git add "results/runs_*_$tag.jsonl" "results/analysis_*_$tag.json" "results/separability_$tag.json" 2>$null
  git commit -q -m "overnight: add $tag runs (factuality+memorization)" 2>> $LOG
  git push -q 2>> $LOG
  Pop-Location
}

# Final combined analysis writeup
& $PY (Join-Path $Study "src\combine_results.py") *>> $LOG
Push-Location $Study
git add RESULTS_multimodel.md 2>$null
git commit -q -m "overnight: combined multi-model results" 2>> $LOG
git push -q 2>> $LOG
Pop-Location
Log "=== overnight sweep done ==="
