# Section 4.5 (Table 16): sensitivity of MOPSO to the HV_pt reference coordinate rho of Eq. (14) (search guidance only; the HV performance metric
# always uses the reference point (1.1, 1.1)). Ceiling fan, E_max = 20,000, seeds 4001-4030.
. "$PSScriptRoot\config.ps1"
$dir = Join-Path $Out 'rho'
New-Item -ItemType Directory -Force $dir | Out-Null
$set = Format-Set (Get-BestParams 'MOPSO')
$inv = [Globalization.CultureInfo]::InvariantCulture
foreach ($r in $RhoLevels) {
    $rs = ([double]$r).ToString('R', $inv)
    Invoke-SsdspOnce (Join-Path $dir "rho_${TuneCase}_$rs") @('--alg', 'MOPSO', '--case', $TuneCase,
        '--budget', "$AblBudget", '--seeds', $Seeds.rho, '--set', $set, '--rho', $rs, '--tag', "rho_$rs",
        '--front', '--threads', "$Threads")
}
