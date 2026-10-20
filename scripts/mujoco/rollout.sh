#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME="${HF_HOME:-$PWD/runtime/huggingface}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"
"$python_bin" -m mujoco_transfer.run --mode rollout --device cuda \
  --scene runtime/panda_ball_reference/scene.xml \
  --config runtime/panda_ball_reference/reference_config.json "$@"
