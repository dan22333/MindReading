#!/bin/bash
# Run a contrast-loss sweep from a run-list file, PAR jobs in parallel on the L4.
#   bash run_contrast.sh contrast_stageA_runs.txt 2
set -u
cd ~/MindReading/brainconcept/vendor_conceptlab
RUNS_FILE="${1:-$HOME/MindReading/brainconcept/contrast_stageA_runs.txt}"
PAR="${2:-2}"
mapfile -t RUNS < "$RUNS_FILE"
echo "=== $(date) contrast sweep: ${#RUNS[@]} runs, $PAR parallel ==="

run_one(){
  local cfg="$1"
  echo "=== $(date) START $cfg ==="
  PYTHONPATH=. ~/braindive_venv/bin/python -m scripts.train \
      --config_path configs/${cfg}.yaml > "$HOME/contrast_${cfg}.log" 2>&1
  echo "=== $(date) DONE $cfg (exit $?) ==="
}

for cfg in "${RUNS[@]}"; do
  while (( $(jobs -r | wc -l) >= PAR )); do sleep 20; done
  run_one "$cfg" &
  sleep 25   # stagger: don't collide two Kandinsky loads on GPU mem
done
wait
echo "=== $(date) CONTRAST SWEEP COMPLETE ($RUNS_FILE) ==="
