#!/usr/bin/env bash
# Ask a running job to stop at its next checkpoint, so nothing is lost.
set -euo pipefail
cd "$(dirname "$0")"
: > STOP_AFTER_CHECKPOINT
echo "Stop requested. The run will finish its current checkpoint and then exit."
echo "Nothing is lost: ./run.sh resumes from that point."
