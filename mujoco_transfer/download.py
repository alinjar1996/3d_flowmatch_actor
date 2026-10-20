"""Download and verify the public checkpoint; publish the file atomically."""

import argparse
from pathlib import Path
import urllib.request

from .policy import CHECKPOINT_SHA256, CHECKPOINT_URL, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-data", action="store_true",
                        help="Download the 229MB test metadata used for reference calibration")
    args = parser.parse_args()
    if args.reference_data:
        url = "https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2_test.zip"
        expected_hash = "d732886a9a9790a51d4ca71bcaa1727662c6dc0c490d402a1b0fce440c29d5a5"
        args.output = args.output or Path("runtime/peract2_test.zip")
    else:
        url, expected_hash = CHECKPOINT_URL, CHECKPOINT_SHA256
        args.output = args.output or Path("checkpoints/3dfa_peract2.pth")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if sha256(args.output) == expected_hash:
            print(f"Verified existing download: {args.output}")
            return
        raise ValueError(f"Existing file differs from verified release: {args.output}")
    temporary = args.output.with_suffix(".download")
    try:
        urllib.request.urlretrieve(url, temporary)
        if sha256(temporary) != expected_hash:
            raise ValueError("Downloaded SHA256 does not match the verified release")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Verified download: {args.output}")


if __name__ == "__main__":
    main()
