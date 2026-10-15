#!/usr/bin/env bash
# Reproduce all Part I results. Models are trained one after another (never concurrently)
# so that the reported training times are not distorted by sharing the GPU.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}

$PY data.py                                  # download PubMed 20k RCT, build subset + vocab + id cache
for m in rnn lstm transformer; do
  $PY train.py --model "$m" --seed 42 2>&1 | tee "results/train_$m.log"
done
$PY evaluate.py                              # n-gram + test perplexity, timing, generations
$PY plot.py                                  # loss curves
