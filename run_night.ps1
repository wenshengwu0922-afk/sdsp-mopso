# Whole pipeline: build + self-tests -> parameter grids + sensitivity -> ablation -> rho sensitivity ->
# formal comparison -> statistics and figures. Log: results_v5\run_log.txt.
# Safe to restart after an interruption: finished ssdsp calls are skipped.
. "$PSScriptRoot\config.ps1"
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force $Out | Out-Null
Start-Transcript -Path (Join-Path $Out 'run_log.txt') -Append | Out-Null
$sw = [Diagnostics.Stopwatch]::StartNew()
try {
    foreach ($s in @('00_build.ps1', '01_grid.ps1', '02_ablation.ps1', '02b_rho.ps1',
                     '03_compare.ps1', '04_analyze.ps1')) {
        Write-Host ("`n=== {0}  ({1:N1} min) ===" -f $s, $sw.Elapsed.TotalMinutes)
        & (Join-Path $PSScriptRoot $s)
    }
    Write-Host ("`nAll steps finished in {0:N1} min. Main outputs: results_v5\grid\tuning_table.md, results_v5\report.md" -f $sw.Elapsed.TotalMinutes)
} finally {
    Stop-Transcript | Out-Null
}
