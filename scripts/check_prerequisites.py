from __future__ import annotations

import shutil
import sys
from pathlib import Path

MIN_FREE_GB = 25
REQUIRED_DIRECTORIES = (
    Path("data/raw"),
    Path("data/staged"),
    Path("data/curated"),
    Path("data/manifests"),
)


def main() -> int:
    missing = [str(path) for path in REQUIRED_DIRECTORIES if not path.is_dir()]
    free_gb = shutil.disk_usage(".").free / 1024**3

    print(f"Python: {sys.version.split()[0]}")
    print(f"Free disk: {free_gb:.1f} GB")
    print(f"Directories present: {not missing}")

    if missing:
        print("Missing directories:", ", ".join(missing))
        return 1
    if free_gb < MIN_FREE_GB:
        print(f"Need at least {MIN_FREE_GB} GB free before bulk acquisition.")
        return 1

    print("Prerequisites passed. API keys are not required for this phase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
