# Experiment design (single source of truth). All runs: population/swarm size 100, rule B,
# termination by the evaluation budget E_max (one evaluation = one sequence scanned at all N stopping points).
$Root    = $PSScriptRoot
$Exe     = Join-Path $Root 'ssdsp.exe'
$Threads = [Environment]::ProcessorCount          # results do not depend on the number of threads

# Small / medium / large budgets E_max per case (Table 8; comparison in Section 4.3)
$Budgets = [ordered]@{ stapler = @(500, 1000, 1500); fan = @(20000, 50000, 100000) }

# Parameter tuning (Section 4.2), ablation (4.4) and rho sensitivity (4.5): ceiling fan, smallest budget of the
# larger case (Table 8). With 100,000 evaluations many MOPSO and PSO settings reach the exact optimum in every
# run, so the tuning criterion would not discriminate.
$TuneCase   = 'fan'
$TuneBudget = 20000
$AblBudget  = 20000

# Disjoint seed sets, 30 independent runs each
$Seeds = [ordered]@{
    grid     = '1001-1030'   # parameter grids
    ablation = '2001-2030'   # ablation study
    rho      = '4001-4030'   # sensitivity to the HV_pt reference coordinate rho
    compare  = '5001-5030'   # comparison (Section 4.3; seeds not used during the design of the method)
}

# Full-factorial grids. MOPSO and PSO use the same grid (same parameters w, c1, c2 and the same levels).
$PsoGrid = 'w=0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9;c1=0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10,12,15,20;c2=0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5'
$Grids = [ordered]@{
    MOPSO = $PsoGrid                                                                                   # 2090 sets
    PSO   = $PsoGrid                                                                                   # 2090 sets
    KG_GA = 'pc=0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.975,1;pm=0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.975,1'  # 1640 sets
    NSGA2 = 'pc=0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.975,1;pm=0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.975,1'  # 1640 sets
    TBGA  = 'team_ratio=0.03,0.05,0.08,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1;delta=0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5,1;nm_ratio=0.25,0.5,1,1.5,2,2.5,3,4,5,6,8,10,15,20'  # 1680 sets
}

# Sensitivity to the HV_pt reference coordinate rho (selects the representative stopping point that the
# personal-best comparison uses; the HV performance metric always uses the reference point (1.1, 1.1))
$RhoLevels = @(1.01, 1.05, 1.1, 1.2, 1.5, 2, 5)

$Trace = 100   # convergence checkpoints per run (every 1 % of the budget)

$Out = Join-Path $Root 'results_v5'   # all outputs of this experiment version

function Invoke-Ssdsp([string[]]$ArgList) {
    & $Exe @ArgList
    if ($LASTEXITCODE -ne 0) { throw "ssdsp failed: $($ArgList -join ' ')" }
}

# Runs one ssdsp call unless its <out>_runs.csv already exists (the driver writes that file last, so an existing
# file means the call finished). Re-running a script therefore resumes after an interruption.
function Invoke-SsdspOnce([string]$OutPrefix, [string[]]$ArgList) {
    if (Test-Path "${OutPrefix}_runs.csv") { Write-Host "skip (done): $OutPrefix"; return }
    Invoke-Ssdsp ($ArgList + @('--out', $OutPrefix))
}

function Get-BestParams([string]$Alg) {
    $file = Join-Path $Out 'grid\best_params.json'
    if (-not (Test-Path $file)) { throw "best_params.json not found: run 01_grid.ps1 first" }
    $b = Get-Content $file -Raw | ConvertFrom-Json
    if (-not $b.PSObject.Properties[$Alg]) { throw "no best parameters for $Alg in best_params.json" }
    $h = [ordered]@{}
    foreach ($p in $b.$Alg.PSObject.Properties) { $h[$p.Name] = [double]$p.Value }
    return $h
}

function Format-Set($Params) {
    $inv = [Globalization.CultureInfo]::InvariantCulture
    return ($Params.Keys | ForEach-Object { '{0}={1}' -f $_, ([double]$Params[$_]).ToString('R', $inv) }) -join ','
}

function Get-Python {
    foreach ($c in @('python', 'py')) { if (Get-Command $c -ErrorAction SilentlyContinue) { return $c } }
    throw 'Python not found (needs numpy, pandas, scipy; matplotlib for figures)'
}

function Invoke-Python([string[]]$ArgList) {
    $py = Get-Python
    & $py @ArgList
    if ($LASTEXITCODE -ne 0) { throw "python failed: $($ArgList -join ' ')" }
}
