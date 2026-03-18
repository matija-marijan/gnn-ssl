#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gnn-ssl

ROOT_DIR="/home/matija-marijan/Projects/gnn-ssl"
SYDRA_DIR="${ROOT_DIR}/sydra"
DATA_ROOT="${ROOT_DIR}/datasets/full"

mkdir -p "${DATA_ROOT}"

pushd "${SYDRA_DIR}" >/dev/null

echo "Generating training set (15k split into 5m and 7m)"
python main.py --config-path "${ROOT_DIR}/config" --config-name sydra_train_5_7 \
  dataset_dir="${DATA_ROOT}/train_5m" \
  mics.n_mics=5 \
  n_samples=7500

python main.py --config-path "${ROOT_DIR}/config" --config-name sydra_train_5_7 \
  dataset_dir="${DATA_ROOT}/train_7m" \
  mics.n_mics=7 \
  n_samples=7500 \
  random_seed=43

echo "Generating validation set (5k split into 5m and 7m)"
python main.py --config-path "${ROOT_DIR}/config" --config-name sydra_train_5_7 \
  dataset_dir="${DATA_ROOT}/val_5m" \
  mics.n_mics=5 \
  n_samples=2500 \
  random_seed=44

python main.py --config-path "${ROOT_DIR}/config" --config-name sydra_train_5_7 \
  dataset_dir="${DATA_ROOT}/val_7m" \
  mics.n_mics=7 \
  n_samples=2500 \
  random_seed=45

echo "Generating test sets (10k total over 4/5/6/7 mics)"
python main.py --config-path "${ROOT_DIR}/config" --config-name sydra_train_5_7 \
  dataset_dir="${DATA_ROOT}/test_4m" \
  mics.n_mics=4 \
  n_samples=2500 \
  random_seed=46

python main.py --config-path "${ROOT_DIR}/config" --config-name sydra_train_5_7 \
  dataset_dir="${DATA_ROOT}/test_5m" \
  mics.n_mics=5 \
  n_samples=2500 \
  random_seed=47

python main.py --config-path "${ROOT_DIR}/config" --config-name sydra_train_5_7 \
  dataset_dir="${DATA_ROOT}/test_6m" \
  mics.n_mics=6 \
  n_samples=2500 \
  random_seed=48

python main.py --config-path "${ROOT_DIR}/config" --config-name sydra_train_5_7 \
  dataset_dir="${DATA_ROOT}/test_7m" \
  mics.n_mics=7 \
  n_samples=2500 \
  random_seed=49

popd >/dev/null
