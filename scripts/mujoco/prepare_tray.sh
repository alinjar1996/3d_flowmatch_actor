#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python_bin="${PYTHON_BIN:-.venv/bin/python}"
"$python_bin" -m mujoco_transfer.download --source-scenes --task bimanual_lift_tray
bash scripts/mujoco/prepare_visuals.sh --task bimanual_lift_tray \
  --output runtime/source_visuals_tray
"$python_bin" -m mujoco_transfer.tray "$@"
