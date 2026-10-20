"""Export the Panda task from a git ref without checking out or changing it."""

import argparse
import io
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

ASSET_PATH = "real_demo/panda_mjx/franka_emika_panda"
DEFAULT_REF = "origin/issue_38_lift_ball_panda"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def prepare(repo, ref, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    commit = git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
    archive = git(repo, "archive", "--format=zip", commit, ASSET_PATH)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            relative = Path(info.filename).relative_to(ASSET_PATH)
            target = output / relative
            if not target.resolve().is_relative_to(output):
                raise ValueError("Unexpected archive path")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(info))

    robot = ET.parse(output / "panda_scene.xml")
    root = robot.getroot()
    actuators = ET.SubElement(root, "actuator")
    for side in ("left", "right"):
        for i in range(1, 8):
            joint_name = f"joint{i}_{side}"
            joint = root.find(f".//joint[@name='{joint_name}']")
            joint_range = joint.get("range", "-2.8973 2.8973")
            ET.SubElement(actuators, "position", name=f"drive_{joint_name}",
                          joint=joint_name, kp="600", kv="50",
                          ctrlrange=joint_range, forcerange="-87 87" if i < 5 else "-12 12")
        for finger in ("left", "right"):
            name = f"{finger}_finger_{side}"
            body = root.find(f".//body[@name='{name}']")
            # Original branch has rigid fingers at +/-0.04. Replace with sliders.
            body.set("pos", "0 0 0.0584")
            ET.SubElement(body, "joint", name=name, type="slide", axis="0 1 0",
                          range="0 0.04", damping="2", armature="0.01")
            ET.SubElement(actuators, "position", name=f"drive_{name}", joint=name,
                          kp="300", kv="10", ctrlrange="0 0.04", forcerange="-20 20")
        hand = root.find(f".//body[@name='hand_{side}']")
        # Optical -Z points along hand +Z; mounted behind the fingertip plane.
        ET.SubElement(hand, "camera", name=f"wrist_{side}", pos="0 -0.10 0.035",
                      xyaxes="1 0 0 0 -0.861934 0.507020", fovy="60")
    robot.write(output / "panda_scene.xml")

    scene = ET.parse(output / "scene.xml")
    root = scene.getroot()
    option = root.find("option")
    option.set("timestep", "0.002")
    option.set("iterations", "50")
    option.set("integrator", "implicitfast")
    flag = option.find("flag")
    if flag is not None:
        option.remove(flag)
    ET.SubElement(root.find("worldbody"), "camera", name="front",
                  pos="1.4 0 1.5", xyaxes="0 1 0 -0.393919 0 0.919145", fovy="60")
    scene.write(output / "scene.xml")
    manifest = {
        "source_repo": str(Path(repo).resolve()), "source_ref": ref,
        "source_commit": commit, "asset_path": ASSET_PATH,
        "changes": ["four actuated finger sliders", "14 arm position actuators",
                    "front and two wrist RGB-D cameras", "2ms implicitfast dynamics"],
        "calibration_status": "provisional; source RLBench frame and TCP need measurement",
    }
    (output / "source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output / "scene.xml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--output", type=Path, default=Path("runtime/panda_ball"))
    args = parser.parse_args()
    print(prepare(args.repo, args.ref, args.output))


if __name__ == "__main__":
    main()
