"""Phase A2 — Clean-room data asset importer with SHA-256 checksums and provenance logging.

Per TechGAR.md Phase A:
- Strictly imports dataset/video/calibration assets into the target directory.
- Rejects any source code (.py, .sh, .bat) to maintain clean-room separation.
- Calculates and logs SHA-256 checksums for auditability in provenance.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".json", ".txt", ".csv", ".yaml", ".yml", ".png", ".jpg"}
FORBIDDEN_EXTENSIONS = {".py", ".pyc", ".pyd", ".sh", ".bat", ".cmd", ".exe", ".ps1"}


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file in chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def import_asset(src_path: Path, dst_dir: Path) -> dict:
    """Import a single asset file into dst_dir with validation."""
    if not src_path.is_file():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    if src_path.suffix.lower() in FORBIDDEN_EXTENSIONS:
        raise ValueError(f"Forbidden extension {src_path.suffix} - only data assets can be imported!")
    if src_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported extension {src_path.suffix}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / src_path.name
    shutil.copy2(src_path, dst_path)

    sha = compute_sha256(dst_path)
    return {
        "filename": src_path.name,
        "source": str(src_path.resolve()),
        "destination": str(dst_path.resolve()),
        "size_bytes": dst_path.stat().st_size,
        "sha256": sha,
        "imported_at": time.time(),
    }


def import_directory(src_dir: Path, dst_dir: Path) -> list[dict]:
    """Import all allowed assets from src_dir into dst_dir."""
    if not src_dir.is_dir():
        raise NotADirectoryError(f"Source directory not found: {src_dir}")

    results = []
    for p in src_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS and p.suffix.lower() not in FORBIDDEN_EXTENSIONS:
            rel = p.relative_to(src_dir)
            target = dst_dir / rel.parent
            record = import_asset(p, target)
            results.append(record)

    provenance_path = dst_dir / "provenance.json"
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump({"imported_count": len(results), "records": results}, f, indent=2)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean-room data asset importer for TechGAR")
    parser.add_argument("--src", required=True, help="Path to source directory or file")
    parser.add_argument("--dst", required=True, help="Destination directory in TechGAR")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if src.is_file():
        record = import_asset(src, dst)
        print(f"Imported single asset: {record['filename']} (SHA256: {record['sha256'][:12]}...)")
    elif src.is_dir():
        records = import_directory(src, dst)
        print(f"Imported {len(records)} assets into {dst}")
    else:
        print(f"Error: {src} not found", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
