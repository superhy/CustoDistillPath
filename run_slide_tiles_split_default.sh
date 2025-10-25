#!/usr/bin/env bash

# Exit immediately if any command fails.
set -euo pipefail

python run_process.py split-tiles \
  --slide-dir "/exafs1/well/rittscher/projects/TCGA-COAD/data/slides" \
  --tile-pkl-dir "/exafs1/well/rittscher/projects/TCGA-COAD/data/tilelists"
