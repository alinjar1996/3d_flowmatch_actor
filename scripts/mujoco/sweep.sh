#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME="${HF_HOME:-$PWD/runtime/huggingface}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"
exec "$python_bin" -m mujoco_transfer.sweep "$@"
