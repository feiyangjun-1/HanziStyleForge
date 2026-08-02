#!/usr/bin/env bash
# Start or resume the complete run on Linux or macOS.
# Safe to interrupt: every stage and every generated glyph is checkpointed,
# and running this again picks up where it stopped.
set -uo pipefail
cd "$(dirname "$0")"

CONFIG="${HSF_CONFIG:-config_months_12gb.json}"
MAX_ATTEMPTS=20

[ -x .venv/bin/python ] || { echo "Run ./install.sh first."; exit 1; }
# A stop request left over from a previous session would otherwise halt this
# run at its first checkpoint.
rm -f STOP_AFTER_CHECKPOINT

attempt=0
while :; do
  attempt=$((attempt + 1))
  echo
  echo "============================================================"
  echo " HanziStyleForge Fusion - attempt $attempt of $MAX_ATTEMPTS"
  echo " Interrupting is safe. Run this script again to resume."
  echo "============================================================"
  .venv/bin/python hanzistyleforge.py --config "$CONFIG" fusion-auto-months
  code=$?
  case $code in
    0)
      echo
      echo "Finished. Your font is in build/"
      exit 0 ;;
    75)
      echo
      echo "Stopped safely after a checkpoint. Run this script again to resume."
      exit 75 ;;
    76)
      echo
      echo "Stopped by the style-collapse guard. Read"
      echo "DIFFUSION_STYLE_COLLAPSE_DETECTED.json before changing any setting."
      exit 76 ;;
    130)
      echo
      echo "Stopped by Ctrl+C. Run this script again to resume."
      exit 130 ;;
  esac
  if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
    echo
    echo "Failed $MAX_ATTEMPTS times in a row. This is a persistent fault, not a"
    echo "temporary one, so retrying further would only repeat it. Fix the cause"
    echo "shown above and run this script again."
    exit 1
  fi
  echo
  echo "Exited with code $code. Retrying in 60 seconds..."
  sleep 60
done
