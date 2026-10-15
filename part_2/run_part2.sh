#!/usr/bin/env bash
# Reproduce all Part II results. Run AFTER part_1/run_part1.sh (setting A reuses part_1/checkpoints/lstm.pt).
# Runs are sequential (never concurrent with other training) so the reported times are not distorted.
# CONTROLS=1 bash part_2/run_part2.sh   also trains the two optional 2x2 controls (~30 min more).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export PYTHONHASHSEED=0  # gensim seeds word vectors with Python's string hash

GLOVE=emb/glove-wiki-gigaword-100.gz
if [ ! -f "$GLOVE" ]; then
  curl -L -o "$GLOVE" \
    https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz
fi

$PY train_w2v.py      # setting B embeddings (timed separately from LM training)
$PY build_glove.py    # setting C embeddings + coverage report

cd ../part_1
test -f checkpoints/lstm.pt || { echo "missing part_1/checkpoints/lstm.pt: run part_1 first"; exit 1; }
$PY train.py --model lstm --emb ../part_2/emb/w2v100.npy   --freeze_emb --tag lstm_w2v   --seed 42 2>&1 | tee results/train_lstm_w2v.log
$PY train.py --model lstm --emb ../part_2/emb/glove100.npy --freeze_emb --tag lstm_glove --seed 42 2>&1 | tee results/train_lstm_glove.log
if [ "${CONTROLS:-0}" = "1" ]; then
  $PY train.py --model lstm --freeze_emb --tag lstm_randfrozen --seed 42 2>&1 | tee results/train_lstm_randfrozen.log
  $PY train.py --model lstm --emb ../part_2/emb/glove100.npy --tag lstm_glove_ft --seed 42 2>&1 | tee results/train_lstm_glove_ft.log
fi

cd ../part_2
$PY evaluate_part2.py
