# Section 4.2: full-factorial parameter grids of all five algorithms (ceiling fan, E_max = 20,000, seeds 1001-1030),
# parameter selection (Table 11) and factorial ANOVA of the MOPSO grid (Table 12).
. "$PSScriptRoot\config.ps1"
$dir = Join-Path $Out 'grid'
New-Item -ItemType Directory -Force $dir | Out-Null
foreach ($alg in $Grids.Keys) {
    Invoke-SsdspOnce (Join-Path $dir "grid_${alg}_$TuneCase") @('--alg', $alg, '--case', $TuneCase,
        '--budget', "$TuneBudget", '--seeds', $Seeds.grid, '--grid', $Grids[$alg], '--tag', 'grid', '--threads', "$Threads")
}
Invoke-Python @((Join-Path $Root 'analysis\analyze_grid.py'), $dir)
