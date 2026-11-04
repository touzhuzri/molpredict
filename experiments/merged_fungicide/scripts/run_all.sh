#!/usr/bin/env bash
set -euo pipefail
cd /d/works/yuce
PY=./.venv/Scripts/python.exe
OUT=experiments/merged_fungicide

$PY $OUT/scripts/build_dataset.py
$PY $OUT/scripts/make_splits.py
$PY $OUT/scripts/prepare_features.py
$PY $OUT/scripts/train_models.py --experiments A,B,C \
  --split-seeds 42 --seeds 42,43,44
$PY $OUT/scripts/train_models.py --experiments B_OLD_ONLY \
  --split-seeds 42 --seeds 42,43,44
$PY $OUT/scripts/analyze_results.py
$PY $OUT/scripts/padel_permutation.py --experiment B \
  --split-seed 42 --seed 42
$PY $OUT/scripts/make_figures.py
$PY $OUT/scripts/make_manifest.py
