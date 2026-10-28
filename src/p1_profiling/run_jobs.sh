#!/usr/bin/env bash
# Run PyTorchSim timing jobs in parallel inside the torchsim container.
#   bash src/p1_profiling/run_jobs.sh <jobs.txt> <status.log> [parallelism=2]
# Each line of jobs.txt: <phase|stage|replay> <target> <config.yml>, e.g. "stage e2e configs/pytorchsim/edge_32x32_lpddr5.yml".
# Output goes to results/<config stem>/torchsim/<phase>/timing/ or results/<config stem>/torchsim_e2e/<stage>/timing/;
# "replay" re-runs a finished stage's trace at instruction level (replay_trace.py) and writes into that same dir.
# Put the longest jobs (e2e, vision) first. Budget ~2-3 GB per Python runner plus TOGSim (up to ~11 GB for
# edge 32x32 vision/e2e, ~2-3 GB for 64x64): see CLAUDE.md "Memory and CPU".
# A watchdog kills this queue's simulations if the HOST's MemAvailable drops below $MIN_AVAIL_MB (default
# 1500): the container's /proc/meminfo shows host memory, and a swap-thrashing host froze the laptop on 10-23.
set -u
jobs=$1; status=$2; par=${3:-2}
cd /workspace/3-MDP
: > "$status"
( while sleep 5; do
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$avail" -lt "${MIN_AVAIL_MB:-1500}" ]; then
      echo "WATCHDOG host MemAvailable ${avail} MB: killing simulations" >> "$status"
      pkill -f torchsim_e2e.py; pkill -f torchsim_smolvla.py; pkill -f replay_trace.py; pkill -x Simulator; pkill -x xargs; exit
    fi
  done ) &
wd=$!
run() {
  kind=$1; target=$2; config=$3; status=$4
  root=results/$(basename "$config" .yml)
  if [ "$kind" = replay ]; then
    python src/p1_profiling/replay_trace.py "$root/torchsim_e2e/$target/timing" > "$root/torchsim_e2e/$target/timing/replay_summary.txt" 2>&1
  elif [ "$kind" = phase ]; then
    rm -rf "$root/torchsim/$target/timing"
    python src/p1_profiling/torchsim_smolvla.py "$target" --mode timing --config "$config" --outdir "$root/torchsim" > /dev/null 2>&1
  else
    rm -rf "$root/torchsim_e2e/$target/timing"
    python src/p1_profiling/torchsim_e2e.py "$target" --config "$config" --outdir "$root/torchsim_e2e" > /dev/null 2>&1
  fi
  rc=$?
  echo "$kind $target $(basename "$config" .yml) exit $rc" >> "$status"
}
export -f run
grep -v '^\s*\(#\|$\)' "$jobs" | xargs -P "$par" -L 1 bash -c 'run "$0" "$1" "$2" '"$status"
kill $wd 2>/dev/null
echo ALL_DONE >> "$status"
