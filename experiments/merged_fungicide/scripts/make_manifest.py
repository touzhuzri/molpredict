"""Hash inputs, processed data, splits, features and code."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DATA, FEATURES, NEW_ELIGIBLE, NEW_EVIDENCE, NEW_LABELS, NEW_MOLECULES,
    OLD_LABELED, OUT, SPLITS, dump_json, sha256,
)


def main() -> int:
    source = {
        "old_ppdb_labeled": {"path": str(OLD_LABELED),
                             "sha256": sha256(OLD_LABELED)},
        "new_molecules": {"path": str(NEW_MOLECULES),
                          "sha256": sha256(NEW_MOLECULES)},
        "new_labels": {"path": str(NEW_LABELS),
                       "sha256": sha256(NEW_LABELS)},
        "new_evidence": {"path": str(NEW_EVIDENCE),
                         "sha256": sha256(NEW_EVIDENCE)},
        "new_eligible": {"path": str(NEW_ELIGIBLE),
                         "sha256": sha256(NEW_ELIGIBLE)},
    }
    data_files = {}
    for path in sorted(DATA.glob("*")):
        if path.is_file() and path.name != "data_manifest.json":
            data_files[path.name] = {
                "sha256": sha256(path), "bytes": path.stat().st_size,
            }
    split_files = {
        p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in sorted(SPLITS.glob("*"))
    }
    feature_files = {
        p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in sorted(FEATURES.glob("*")) if p.is_file()
    }
    code_files = {
        p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in sorted((OUT / "scripts").glob("*.py"))
    }
    payload = {
        "root": str(ROOT),
        "sources": source,
        "data": data_files,
        "splits": split_files,
        "features": feature_files,
        "scripts": code_files,
    }
    dump_json(DATA / "data_manifest.json", payload)
    dump_json(OUT / "logs" / "code_manifest.json", code_files)
    print(json.dumps({
        "sources": len(source), "data": len(data_files),
        "splits": len(split_files), "features": len(feature_files),
        "scripts": len(code_files),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
