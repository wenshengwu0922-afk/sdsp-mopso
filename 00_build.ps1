# Build ssdsp.exe with g++ (C++17) and run self-tests. Tries: multithreaded static -> multithreaded -> single-threaded.
. "$PSScriptRoot\config.ps1"
Push-Location $Root
try {
    $variants = @(
        @('-std=c++17', '-O2', '-static', '-pthread', 'src/main.cpp', '-o', 'ssdsp.exe'),
        @('-std=c++17', '-O2', '-pthread', 'src/main.cpp', '-o', 'ssdsp.exe'),
        @('-std=c++17', '-O2', '-DSSDSP_NO_THREADS', 'src/main.cpp', '-o', 'ssdsp.exe')
    )
    $ok = $false
    foreach ($v in $variants) {
        & g++ @v
        if ($LASTEXITCODE -eq 0) { Write-Host "Built with: g++ $($v -join ' ')"; $ok = $true; break }
    }
    if (-not $ok) { throw 'build failed (is g++ with C++17 support on PATH?)' }

    # Self-test 1: exact fronts (enumeration of all stopping states); also refreshes analysis\exact_*.csv
    $expect = @{ stapler = 'HV = 0.831917'; fan = 'HV = 0.859667' }
    foreach ($case in @('stapler', 'fan')) {
        $msg = & .\ssdsp.exe --exact --case $case --out (Join-Path $Root "analysis\exact_$case")
        if ($LASTEXITCODE -ne 0 -or ($msg -notmatch [regex]::Escape($expect[$case]))) { throw "self-test (exact $case) failed: $msg" }
        Write-Host $msg
    }
    # Self-test 2: random search reaches the stapler optimum with 20,000 evaluations
    $msg = & .\ssdsp.exe --alg RS --case stapler --budget 20000 --seeds 1-3 --threads 2 --out (Join-Path $env:TEMP 'ssdsp_selftest')
    if ($LASTEXITCODE -ne 0 -or ($msg -notmatch 'meanHV=0\.831917')) { throw "self-test (RS stapler) failed: $msg" }
    Write-Host $msg
    # Self-test 3: MOPSO runs and uses exactly the budget (the driver stops with an error otherwise)
    $msg = & .\ssdsp.exe --alg MOPSO --case fan --budget 2000 --seeds 1-2 --set 'w=0,c1=5,c2=3' `
        --threads 2 --out (Join-Path $env:TEMP 'ssdsp_selftest')
    if ($LASTEXITCODE -ne 0) { throw "self-test (MOPSO fan) failed: $msg" }
    Write-Host $msg
    Write-Host 'Build OK, self-tests passed.'
} finally { Pop-Location }
