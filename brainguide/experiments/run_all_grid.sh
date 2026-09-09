#!/usr/bin/env bash
# Full 5-samples-per-ROI run, sequential (one GPU): Plan 4 first (fast), then the
# guided grid (baseline + plan1 + plan2 ctrl/tt). Detach with nohup+setsid.
#   nohup setsid bash ~/MindReading/brainguide/experiments/run_all_grid.sh >~/grid.log 2>&1 </dev/null &
set -uo pipefail
cd ~/MindReading
export SEEDS="${SEEDS:-0 1 2 3 4}"   # seeds per ROI; override e.g. SEEDS="0 1 2"
echo "SEEDS=$SEEDS"

echo "############## PLAN 4 (bridge + IP-Adapter): 5 seeds/ROI ##############"
bash brainguide/experiments/test4_dino_clip_bridge/run_test4.sh

echo "############## GUIDED GRID (baseline + plan1 + plan2): 5 seeds/ROI ##############"
bash brainguide/experiments/exp_grid.sh

echo "############## ALL GRID RUNS DONE ##############"
