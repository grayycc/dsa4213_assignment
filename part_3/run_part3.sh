#!/usr/bin/env bash
# Reproduce all Part III results. Run AFTER part_1/run_part1.sh and part_2/run_part2.sh (reuses their checkpoints).
# Language models stay frozen; only logistic-regression classifiers are trained.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}

$PY data_cls.py       # leak-free classifier splits (abstracts unused by the LMs) + label distribution
$PY extract.py        # frozen-LM sentence features for every available checkpoint (missing ones are skipped)
$PY classify.py       # same LR classifier on every representation, bootstrap CIs, error analysis, figures
