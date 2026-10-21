"""Download and verify the public checkpoint; publish the file atomically."""

import argparse
import json
from pathlib import Path
import urllib.request

from .policy import CHECKPOINT_SHA256, CHECKPOINT_URL, sha256


def download_verified(url, expected_hash, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if sha256(output) == expected_hash:
            print(f"Verified existing download: {output}")
            return
        raise ValueError(f"Existing file differs from verified release: {output}")
    temporary = output.with_suffix(".download")
    try:
        urllib.request.urlretrieve(url, temporary)
        if sha256(temporary) != expected_hash:
            raise ValueError("Downloaded SHA256 does not match the verified release")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Verified download: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    kind = parser.add_mutually_exclusive_group()
    kind.add_argument("--reference-data", action="store_true",
                      help="Download the 229MB test metadata used for reference calibration")
    kind.add_argument("--source-scenes", action="store_true",
                      help="Download pinned native scene assets; --output is a directory")
    args = parser.parse_args()
    if args.source_scenes:
        source = json.loads((Path(__file__).parent / "configs/peract2_scene.json").read_text())["source"]
        destination = args.output or Path("runtime/source-audit/RLBench")
        for key in ("scene", "task_model", "task_code"):
            item = source[key]
            url = f"https://raw.githubusercontent.com/markusgrotz/RLBench/{source['commit']}/{item['path']}"
            download_verified(url, item["sha256"], destination / item["path"])
        (destination / "source.json").write_text(json.dumps(source, indent=2) + "\n")
        return
    if args.reference_data:
        url = "https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2_test.zip"
        expected_hash = "d732886a9a9790a51d4ca71bcaa1727662c6dc0c490d402a1b0fce440c29d5a5"
        args.output = args.output or Path("runtime/peract2_test.zip")
    else:
        url, expected_hash = CHECKPOINT_URL, CHECKPOINT_SHA256
        args.output = args.output or Path("checkpoints/3dfa_peract2.pth")
    download_verified(url, expected_hash, args.output)


if __name__ == "__main__":
    main()
