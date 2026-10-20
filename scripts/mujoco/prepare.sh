#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
repo="${1:-/home/aks-lab/colcon_ws/src/manipulator_mujoco}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"

"$python_bin" -m mujoco_transfer.download
"$python_bin" -m mujoco_transfer.download --reference-data
"$python_bin" -m mujoco_transfer.prepare --repo "$repo" \
  --ref origin/issue_38_lift_ball_panda --output runtime/panda_ball_reference
"$python_bin" -m mujoco_transfer.reference --archive runtime/peract2_test.zip \
  --scene runtime/panda_ball_reference/scene.xml \
  --output-config runtime/panda_ball_reference/reference_config.json
"$python_bin" -m mujoco_transfer.verify_settings
