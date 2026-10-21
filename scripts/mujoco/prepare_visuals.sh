#!/usr/bin/env bash
# Export source appearance once; ordinary source_mujoco inference uses only MuJoCo.
# Extra arguments go to the exporter, e.g. --output /tmp/source-visuals-check.
set -euo pipefail
cd "$(dirname "$0")/../.."

python_bin="${PYTHON_BIN:-.venv/bin/python}"
source_sim_root="${COPPELIASIM_ROOT:-$PWD/runtime/source-audit/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}"
if [[ ! -x "$python_bin" ]]; then
  echo "Python environment missing: $python_bin. Prepare the repository .venv or set PYTHON_BIN." >&2
  exit 1
fi
if [[ ! -f "$source_sim_root/libcoppeliaSim.so" ]]; then
  echo "CoppeliaSim 4.1.0 is required for this one-time export. Set COPPELIASIM_ROOT to its installed directory." >&2
  exit 1
fi

export COPPELIASIM_ROOT="$source_sim_root"
export LD_LIBRARY_PATH="$source_sim_root${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_PLUGIN_PATH="$source_sim_root"
export QT_QPA_PLATFORM_PLUGIN_PATH="$source_sim_root/platforms"
export QT_QPA_PLATFORM=offscreen
export PYTHONPATH=""

if ! "$python_bin" -c 'import pyrep' ; then
  echo "The source audit PyRep installation is missing or cannot load CoppeliaSim. Install the markusgrotz/PyRep fork into this Python environment." >&2
  exit 1
fi

"$python_bin" -m mujoco_transfer.download --source-scenes
"$python_bin" scripts/mujoco/export_source_visuals.py \
  --rlbench-root runtime/source-audit/RLBench --output runtime/source_visuals "$@"
