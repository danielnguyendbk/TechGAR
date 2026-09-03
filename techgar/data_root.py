"""Phase A1 — Data root resolution and asset availability checking.

Per TechGAR.md Phase A:
- All data paths are resolved relative to TECHGAR_DATA_ROOT environment variable
  or default to the repository workspace root.
- If video or timestamp files are not present on disk, datasets are flagged
  as 'unavailable' so tests and CLI tools do not silently crash.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_data_root() -> Path:
    """Return the base path for all TechGAR datasets and site assets."""
    env_root = os.environ.get("TECHGAR_DATA_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # Default: repository root (parent of 'techgar' module)
    return Path(__file__).resolve().parents[1]


def resolve_data_path(path: str | Path, base: Path | None = None) -> Path:
    """Resolve a relative or absolute dataset path against the data root."""
    p = Path(path)
    if p.is_absolute():
        return p
    root = base or get_data_root()
    return (root / p).resolve()


def check_dataset_available(dataset: dict, base: Path | None = None) -> bool:
    """Check whether all raw videos and timestamp files declared in a manifest dataset exist."""
    root = base or get_data_root()
    directory = root / dataset.get("directory", "")
    if not directory.is_dir():
        return False

    # Check timestamps
    ts_file = dataset.get("timestamps")
    if ts_file and not (directory / ts_file).is_file():
        return False

    # Check raw videos
    raw_videos = dataset.get("raw_videos", {})
    if not raw_videos:
        return False
    for _, filename in raw_videos.items():
        if not (directory / filename).is_file():
            return False

    return True
