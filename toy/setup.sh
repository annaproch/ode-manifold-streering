#!/usr/bin/env bash
# Set up the toy: fetch ODESteer at a pinned commit (it has no license file, so it is not vendored)
# and install CPU dependencies into toy/.venv.
set -euo pipefail
cd "$(dirname "$0")"

ODESTEER_COMMIT=8a3c481d6493ecb3325eea5ef9c448cccfced7eb
if [ ! -d third_party/odesteer ]; then
  git clone -q https://github.com/ZhaoHongjue/odesteer.git third_party/odesteer
fi
git -C third_party/odesteer checkout -q "$ODESTEER_COMMIT"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/pip install -q -r requirements.txt
fi
echo "ready: toy/.venv/bin/python toy/run_all.py"
