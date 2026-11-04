"""Hash the APPf-like artifacts for auditability."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import OUT, sha256  # noqa: E402

APPF = OUT / "appf_like"
EXCLUDE_DIRS = {"demo_cache", "__pycache__", "logs"}
EXCLUDE_NAMES = {"manifest.json"}


def main() -> int:
    files = []
    for p in sorted(APPF.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(APPF)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if p.name in EXCLUDE_NAMES:
            continue
        files.append({
            "path": str(rel).replace("\\", "/"),
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
        })
    payload = {
        "root": str(APPF),
        "n_files": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
    }
    (APPF / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "n_files": payload["n_files"],
        "total_mb": round(payload["total_bytes"] / 1024 ** 2, 2),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
