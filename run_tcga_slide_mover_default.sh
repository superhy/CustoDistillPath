#!/usr/bin/env bash

# Exit immediately if any command fails.
set -euo pipefail

python run_process.py move-slides \
  --original-data-dir "/exafs1/well/rittscher/shared/datasets/TCGA-GI/CR" \
  --output-dir "/exafs1/well/rittscher/projects/TCGA-COAD/data/slides" \
  --barcode-table "metadata/TCGA_COAD/nationwidechildrens.org_clinical_patient_coad.txt" \
  --mode "copy" \
  --barcode-column "bcr_patient_barcode"
