#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec bash scripts/mujoco/sweep.sh \
  --scene runtime/panda_tray_reference/scene.xml \
  --config runtime/panda_tray_reference/reference_config.json \
  --observation-renderer source_mujoco \
  --source-visual-assets runtime/source_visuals_tray \
  --output runtime/seed_sweeps/tray_000_100 "$@"
