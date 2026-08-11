#!/usr/bin/env bash
# Set up HanziStyleForge on Linux or macOS.
# Windows users run install.bat instead.
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '\n[%s] %s\n' "$1" "$2"; }
die() { printf '\nERROR: %s\n' "$1" >&2; exit 1; }

say 1/5 "Looking for Python 3.10 or newer..."
PY=""
for candidate in python3.14 python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,14) else 1)' 2>/dev/null; then
      PY="$candidate"; break
    fi
  fi
done
[ -n "$PY" ] || die "No Python 3.10-3.14 found.
  Linux:  sudo apt install python3 python3-venv     (or your distribution's equivalent)
  macOS:  brew install python@3.12"
say 1/5 "Using $("$PY" --version) at $(command -v "$PY")"

say 2/5 "Creating the .venv virtual environment..."
[ -x .venv/bin/python ] || "$PY" -m venv .venv
[ -x .venv/bin/python ] || die "Could not create .venv. On Debian or Ubuntu install python3-venv first."
VENV=.venv/bin/python

say 3/5 "Installing font and image libraries..."
"$VENV" -m pip install --upgrade --quiet pip setuptools wheel
"$VENV" -m pip install --quiet -r requirements.txt

say 4/5 "Installing PyTorch..."
OS="$(uname -s)"
if [ "$OS" = "Darwin" ]; then
  # macOS has no NVIDIA GPU, so this is the CPU build. Training is not
  # practical here; see the README. The install still lets you inspect fonts,
  # run the self-test and build a font from glyph images you already have.
  "$VENV" -m pip install --quiet torch
  say 4/5 "macOS detected: CPU build installed. Training needs an NVIDIA GPU."
elif command -v nvidia-smi >/dev/null 2>&1; then
  "$VENV" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu130 \
    || "$VENV" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu128
  "$VENV" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' \
    || die "PyTorch installed but cannot see your GPU. Update your NVIDIA driver and run this again."
else
  "$VENV" -m pip install --quiet torch
  say 4/5 "No NVIDIA GPU detected: CPU build installed. Training needs an NVIDIA GPU."
fi

say 5/5 "Running the self-test..."
"$VENV" selftest.py

cat <<'DONE'

Installation finished.

Next:
  1. Put the font whose STYLE you want at   fonts/target.ttf
  2. Put the font whose SHAPES you want at  refs/ref.otf
  3. Run ./verify.sh
  4. Run ./run.sh
DONE
