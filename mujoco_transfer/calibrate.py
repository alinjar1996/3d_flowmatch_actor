"""Fit the world transform from measured corresponding landmarks (no scaling)."""

import argparse
import json
from pathlib import Path

import numpy as np

from .environment import DEFAULT_CONFIG, read_config
from .geometry import fit_rigid_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landmarks", type=Path, required=True,
                        help="NPZ with mujoco_points and policy_points, both Nx3 in metres")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rms", type=float, default=0.01)
    args = parser.parse_args()
    with np.load(args.landmarks, allow_pickle=False) as data:
        transform, rms = fit_rigid_transform(data["mujoco_points"], data["policy_points"])
    if rms > args.max_rms:
        raise ValueError(f"RMS {rms:.6f}m exceeds {args.max_rms}m; check correspondences/base geometry")
    config = read_config(args.config)
    config["T_policy_from_mujoco"] = transform.tolist()
    config["calibration_status"] = (
        f"World landmark fit RMS={rms:.6f}m; TCP, cameras and base placement require separate checks")
    config["world_calibration"] = {"landmarks": str(args.landmarks.resolve()), "rms_metres": rms}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(f"World fit RMS: {rms:.6f}m; wrote {args.output}")


if __name__ == "__main__":
    main()
