#!/usr/bin/env bash
# Reproduce the whole analysis. Usage: bash scripts/run_all.sh   (needs python3 with the packages listed in requirements.txt)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
$PY 01_download_filter.py
$PY 02_profile.py
$PY 03_clean_impute.py
$PY 04_feature_selection.py
$PY 05_build_report.py
