from __future__ import annotations

import shutil
import sys
from pathlib import Path

MIN_FREE_GB = 25
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_DIRECTORIES = (
    Path("data/raw"),
    Path("data/staged"),
    Path("data/curated"),
    Path("data/manifests"),
)


def main(project_root: Path = PROJECT_ROOT) -> int:
    required_directories = tuple(
        project_root / relative_path for relative_path in REQUIRED_DIRECTORIES
    )
    missing = [str(path) for path in required_directories if not path.is_dir()]

    print(f"Python: {sys.version.split()[0]}")
    print(f"Directories present: {not missing}")

    if missing:
        print("Missing directories:", ", ".join(missing))
        return 1

    raw_store_path = project_root / "data/raw"
    free_gb = shutil.disk_usage(raw_store_path).free / 1024**3
    print(f"Raw-store free disk: {free_gb:.1f} GB")
    if free_gb < MIN_FREE_GB:
        print(f"Need at least {MIN_FREE_GB} GB free before bulk acquisition.")
        return 1

    print("Prerequisites passed. API keys are not required for this phase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
