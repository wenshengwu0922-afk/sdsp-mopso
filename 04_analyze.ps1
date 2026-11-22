# Statistics and tables (results_v5\report.md, results_v5\tables) and figures (results_v5\figures).
. "$PSScriptRoot\config.ps1"
Invoke-Python @((Join-Path $Root 'analysis\analyze_results.py'), $Out)
Invoke-Python @((Join-Path $Root 'analysis\figures.py'), $Out)
