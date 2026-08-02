#!/usr/bin/env bash
# Check that everything is set up correctly before starting a long run.
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${HSF_CONFIG:-config_months_12gb.json}"
[ -x .venv/bin/python ] || { echo "Run ./install.sh first."; exit 1; }

echo "Running the self-test..."
.venv/bin/python selftest.py

echo "Checking your fonts and GPU..."
.venv/bin/python hanzistyleforge.py --config "$CONFIG" check

if [ -f work_hanzistyleforge_fusion_months/dataset/index.csv ]; then
  echo "Checking the data-flow contract..."
  .venv/bin/python hanzistyleforge.py --config "$CONFIG" contract
fi

echo
echo "Everything checks out. Start the run with ./run.sh"
