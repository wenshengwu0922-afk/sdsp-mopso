#!/bin/bash
# Section 4.7 (Table 18): 8 carbon-parameter scenarios x 6 methods x 2 instances, Table 11 parameters,
# E_max = 1,500 (stapler) and 20,000 (fan), seeds 6001-6030. Model data overridden by SSDSP_OCF/RCF/PCF/ETA.
cd "$(dirname "$0")"; mkdir -p out
B=../ssdsp      # build first: make (repo root)
declare -A SET=( [MOPSO]="w=0,c1=3.5,c2=4.5" [PSO]="w=0,c1=12,c2=1.5" [KG_GA]="pc=0.75,pm=0.95" [NSGA2]="pc=0,pm=1" [TBGA]="team_ratio=0.08,delta=0.05,nm_ratio=6" )
run_scen () {  # name OCF RCF PCF ETA
  export SSDSP_OCF=$2 SSDSP_RCF=$3 SSDSP_PCF=$4 SSDSP_ETA=$5
  for c in stapler fan; do
    [ $c = stapler ] && bud=1500 || bud=20000
    for a in MOPSO PSO KG_GA NSGA2 TBGA RS; do
      if [ $a = RS ]; then $B --alg $a --case $c --budget $bud --seeds 6001-6030 --threads 2 --out out/$1_${c}_$a >/dev/null
      else $B --alg $a --case $c --budget $bud --seeds 6001-6030 --set "${SET[$a]}" --threads 2 --out out/$1_${c}_$a >/dev/null; fi
    done
    $B --exact --case $c --out out/$1_${c}_exact >/dev/null
  done
}
O=3.27,3.27,9.85,3.562,3.65; R=0.657,0.678,0.844,2.831,0.697; P=0.662,0.912,0.262; E=0.47,0.47,0.51,0.50,0.34
OI=2.66,2.66,14.8,5.3,3.965; RI=0.71,0.71,0.52,1.6,0.697; EU=0.72,0.72,0.57,0.85,0.48
run_scen base $O $R $P $E
run_scen eta08 $O $R $P 0.376,0.376,0.408,0.40,0.272
run_scen eta12 $O $R $P 0.564,0.564,0.612,0.60,0.408
run_scen etaUNEP $O $R $P $EU
run_scen intlmat $OI $RI $P $E
run_scen proc05 $O $R 0.331,0.456,0.131 $E
run_scen proc15 $O $R 0.993,1.368,0.393 $E
run_scen allintl $OI $RI $P $EU
echo done
