#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <lp|sddp> <num_stages> [loop]"
  exit 1
fi

MODE="$1"
NUM_STAGES="$2"
LOOP_MODE="false"

if [[ $# -eq 3 ]]; then
  if [[ "$3" == "loop" ]]; then
    LOOP_MODE="true"
  else
    echo "Invalid third argument: $3"
    echo "Optional third argument may be 'loop' to iterate from 3 to num_stages."
    exit 1
  fi
fi

if [[ "$MODE" != "lp" && "$MODE" != "sddp" ]]; then
  echo "Invalid mode: $MODE"
  echo "Valid modes are: lp, sddp"
  exit 1
fi

if [[ "$NUM_STAGES" -lt 3 ]]; then
  echo "num_stages must be at least 3"
  exit 1
fi

LOG_DIR="output/balance/$MODE"
mkdir -p "$LOG_DIR"

if [[ "$LOOP_MODE" == "true" ]]; then
  for STAGE in $(seq 3 "$NUM_STAGES"); do
    LOG_FILE="$LOG_DIR/run_${STAGE}.log"
    echo "Running $MODE with num_stages=$STAGE -> $LOG_FILE"
    python examples/balance/run.py "$MODE" "$STAGE" 2>&1 | tee "$LOG_FILE"
    echo
  done
else
  LOG_FILE="$LOG_DIR/run_${NUM_STAGES}.log"
  echo "Running $MODE with num_stages=$NUM_STAGES -> $LOG_FILE"
  python examples/balance/run.py "$MODE" "$NUM_STAGES" 2>&1 | tee "$LOG_FILE"
fi
