# Section 4.3 (Tables 13-14, Figs. 4-5): 2 cases x 3 budgets x (5 algorithms + random search), parameters of
# Table 11, seeds 5001-5030. Records HV(FA) every 1 % of the budget (convergence curves) and the final archives.
. "$PSScriptRoot\config.ps1"
$dir = Join-Path $Out 'compare'
New-Item -ItemType Directory -Force $dir | Out-Null
$algs = @('MOPSO', 'PSO', 'KG_GA', 'NSGA2', 'TBGA', 'RS')
$sets = @{}
foreach ($alg in $algs) { if ($alg -ne 'RS') { $sets[$alg] = Format-Set (Get-BestParams $alg) } }
foreach ($case in $Budgets.Keys) {
    foreach ($budget in $Budgets[$case]) {
        foreach ($alg in $algs) {
            $a = @('--alg', $alg, '--case', $case, '--budget', "$budget", '--seeds', $Seeds.compare, '--tag', 'compare',
                   '--trace', "$Trace", '--front', '--threads', "$Threads")
            if ($alg -ne 'RS') { $a += @('--set', $sets[$alg]) }
            Invoke-SsdspOnce (Join-Path $dir "cmp_${case}_${budget}_$alg") $a
        }
    }
}
