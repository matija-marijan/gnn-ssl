#!/usr/bin/env bash
# Upstream-faithful dataset generation (Option A), run sequentially.
#
# One generation run per microphone configuration, all using SYDRA's default
# seed (random_seed=0) and n_jobs=1. n_jobs must stay 1: with joblib/loky,
# n_jobs > 1 spawns workers that do not inherit the parent RNG state, which
# makes generation non-deterministic and the seed ineffective.
#
# Each run is split into train/val/test deterministically by sample index:
#   raw_5m: 0-7499 train_5m | 7500-9999 val_5m | 10000-12499 test_5m
#   raw_7m: 0-7499 train_7m | 7500-9999 val_7m | 10000-12499 test_7m
#   raw_4m: 0-2499 test_4m
#   raw_6m: 0-2499 test_6m
#
# All other settings (rooms, RT60, reflections, VCTK speech, SNR, 500 ms
# frames, single-mic nodes) come from config/sydra_dataset.yaml and follow
# paper Sec. 5.1/5.2.
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gnn-ssl

ROOT_DIR="${GNN_SSL_ROOT:-/mnt/data/Projects/gnn-ssl}"
SYDRA_DIR="${ROOT_DIR}/sydra"
CONFIG_DIR="${ROOT_DIR}/config"
CONFIG_NAME="sydra_dataset"
RAW_ROOT="${ROOT_DIR}/datasets/upstream_raw"
DATA_ROOT="${ROOT_DIR}/datasets/full"

mkdir -p "${RAW_ROOT}"
cd "${SYDRA_DIR}"

python main.py --config-path "${CONFIG_DIR}" --config-name "${CONFIG_NAME}" \
  dataset_dir="${RAW_ROOT}/raw_5m" mics.n_mics=5 n_samples=12500 \
  random_seed=0 n_jobs=1

python main.py --config-path "${CONFIG_DIR}" --config-name "${CONFIG_NAME}" \
  dataset_dir="${RAW_ROOT}/raw_7m" mics.n_mics=7 n_samples=12500 \
  random_seed=0 n_jobs=1

python main.py --config-path "${CONFIG_DIR}" --config-name "${CONFIG_NAME}" \
  dataset_dir="${RAW_ROOT}/raw_4m" mics.n_mics=4 n_samples=2500 \
  random_seed=0 n_jobs=1

python main.py --config-path "${CONFIG_DIR}" --config-name "${CONFIG_NAME}" \
  dataset_dir="${RAW_ROOT}/raw_6m" mics.n_mics=6 n_samples=2500 \
  random_seed=0 n_jobs=1

python "${ROOT_DIR}/scripts/split_dataset.py" \
  --src "${RAW_ROOT}/raw_5m" --dst "${DATA_ROOT}" \
  --split train_5m:0:7500 --split val_5m:7500:10000 --split test_5m:10000:12500

python "${ROOT_DIR}/scripts/split_dataset.py" \
  --src "${RAW_ROOT}/raw_7m" --dst "${DATA_ROOT}" \
  --split train_7m:0:7500 --split val_7m:7500:10000 --split test_7m:10000:12500

python "${ROOT_DIR}/scripts/split_dataset.py" \
  --src "${RAW_ROOT}/raw_4m" --dst "${DATA_ROOT}" --split test_4m:0:2500

python "${ROOT_DIR}/scripts/split_dataset.py" \
  --src "${RAW_ROOT}/raw_6m" --dst "${DATA_ROOT}" --split test_6m:0:2500

echo "Done. Datasets written to ${DATA_ROOT}"
