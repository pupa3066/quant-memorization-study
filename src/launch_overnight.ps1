<#
launch_overnight.ps1 — schedule the sweep to start at a target time, low priority, detached.
Usage:
  .\src\launch_overnight.ps1              # defaults to next 00:15 local
  .\src\launch_overnight.ps1 -Target "00:15"
Runs the sweep in a background job at BelowNormal priority so it won't hamper interactive use.
Logs to overnight.log.
#>
param(
  [string]$Target = "00:15"
)
$Study = Split-Path -Parent $PSScriptRoot
$LOG   = Join-Path $Study "overnight.log"

$now = Get-Date
try { $t = [datetime]::ParseExact($Target, "HH:mm", $null) } catch { Write-Error "bad time '$Target' (use HH:mm)"; exit 2 }
$target = (Get-Date -Hour $t.Hour -Minute $t.Minute -Second 0)
if ($target -le $now) { $target = $target.AddDays(1) }   # already past -> tomorrow
$wait = [int]($target - $now).TotalSeconds

$msg = "[{0}] scheduled: sweep at {1} (in {2}s)" -f (Get-Date -Format HH:mm:ss), ($target.ToString("yyyy-MM-dd HH:mm")), $wait
$msg | Tee-Object -FilePath $LOG -Append

$sweep = Join-Path $PSScriptRoot "overnight_sweep.ps1"
Start-Job -Name "quant-sweep" -ScriptBlock {
  param($wait, $sweep)
  Start-Sleep -Seconds $wait
  # Lower this job's priority so it won't hamper interactive use.
  (Get-Process -Id $PID).PriorityClass = 'BelowNormal'
  & powershell -NoProfile -ExecutionPolicy Bypass -File $sweep
} -ArgumentList $wait, $sweep | Out-Null

$m2 = "[{0}] launcher backgrounded as job 'quant-sweep'. Get-Content -Wait $LOG to watch." -f (Get-Date -Format HH:mm:ss)
$m2 | Tee-Object -FilePath $LOG -Append
