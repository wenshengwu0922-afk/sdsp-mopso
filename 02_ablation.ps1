# Section 4.4 (Table 15, Figs. 6-7): ablation with the final MOPSO parameters (ceiling fan, E_max = 20,000,
# seeds 2001-2030).
# Core 2 x 2 design (evaluation rule x leader source). The manuscript reports full and swarm_leader; rule_A and
# rule_A_swarm are reported only in the response to the reviewers:
#   full          : rule B + leaders from the archive FA (proposed)
#   rule_A        : rule A (only the representative stopping point enters FA) + leaders from FA
#   swarm_leader  : rule B + leader = random non-dominated member of the swarm (no archive guidance)
#   rule_A_swarm  : rule A + swarm leader
# Supplementary variants (alternatives considered during the design of the method):
#   gbest_leader  : rule B + leader = swarm member with the largest HV_pt (single global best)
#   layered_leader: rule B + swarm stratified into Q = 10 layers by d = 0.5 sigma1 + 0.5 sigma2, leader of each
#                   layer = member with the largest HV_pt (clustered-leader design)
#   hv_pbest      : personal best replaced only if HV_pt increases (instead of Pareto dominance)
#   with_mutation : swap mutation with probability 0.2 after each move
. "$PSScriptRoot\config.ps1"
$dir = Join-Path $Out 'ablation'
New-Item -ItemType Directory -Force $dir | Out-Null
$best = Get-BestParams 'MOPSO'
$variants = [ordered]@{
    full           = @{ params = @{};                  extra = @() }
    rule_A         = @{ params = @{};                  extra = @('--rule', 'A') }
    swarm_leader   = @{ params = @{};                  extra = @('--leader', 'swarm_nd') }
    rule_A_swarm   = @{ params = @{};                  extra = @('--rule', 'A', '--leader', 'swarm_nd') }
    gbest_leader   = @{ params = @{};                  extra = @('--leader', 'layer_hv') }
    layered_leader = @{ params = @{ Q = 10 };          extra = @('--leader', 'layer_hv') }
    hv_pbest       = @{ params = @{};                  extra = @('--pbest', 'hv') }
    with_mutation  = @{ params = @{ pm = 0.2 };        extra = @() }
}
foreach ($tag in $variants.Keys) {
    $p = [ordered]@{}; foreach ($k in $best.Keys) { $p[$k] = $best[$k] }
    foreach ($k in $variants[$tag]['params'].Keys) { $p[$k] = $variants[$tag]['params'][$k] }
    Invoke-SsdspOnce (Join-Path $dir "abl_${TuneCase}_$tag") (@('--alg', 'MOPSO', '--case', $TuneCase,
        '--budget', "$AblBudget", '--seeds', $Seeds.ablation, '--set', (Format-Set $p), '--tag', $tag,
        '--trace', "$Trace", '--front', '--threads', "$Threads") + $variants[$tag]['extra'])
}
