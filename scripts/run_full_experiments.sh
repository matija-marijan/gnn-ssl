#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gnn-ssl

ROOT_DIR="/home/matija-marijan/Projects/gnn-ssl"
DATA_ROOT="${ROOT_DIR}/datasets/full"
TRAIN_5M_DIR="${DATA_ROOT}/train_5m"
TRAIN_7M_DIR="${DATA_ROOT}/train_7m"
VAL_5M_DIR="${DATA_ROOT}/val_5m"
VAL_7M_DIR="${DATA_ROOT}/val_7m"

TEST_4M_DIR="${DATA_ROOT}/test_4m"
TEST_5M_DIR="${DATA_ROOT}/test_5m"
TEST_6M_DIR="${DATA_ROOT}/test_6m"
TEST_7M_DIR="${DATA_ROOT}/test_7m"

CKPT_DIR="${ROOT_DIR}/trained_models/full"
mkdir -p "${CKPT_DIR}"

pushd "${ROOT_DIR}" >/dev/null

echo "[1/5] Training GNN-GCC"
python train.py --config-name experiment_full_train \
  +model@_here_=gnn_gcc \
  'inputs_train.dataset_paths.training=["'"${TRAIN_5M_DIR}"'","'"${TRAIN_7M_DIR}"'"]' \
  'inputs_train.dataset_paths.validation=["'"${VAL_5M_DIR}"'","'"${VAL_7M_DIR}"'"]' \
  inputs_train.dataset_paths.test="${TEST_5M_DIR}" \
  hydra.run.dir="${CKPT_DIR}/train_gnn_gcc"

echo "[2/5] Training GNN-SLF"
python train.py --config-name experiment_full_train \
  +model@_here_=gnn_slf \
  'inputs_train.dataset_paths.training=["'"${TRAIN_5M_DIR}"'","'"${TRAIN_7M_DIR}"'"]' \
  'inputs_train.dataset_paths.validation=["'"${VAL_5M_DIR}"'","'"${VAL_7M_DIR}"'"]' \
  inputs_train.dataset_paths.test="${TEST_5M_DIR}" \
  hydra.run.dir="${CKPT_DIR}/train_gnn_slf"

echo "[3/5] Training GraphRelNet"
python train.py --config-name experiment_full_train \
  +model@_here_=graph_rel_net \
  'inputs_train.dataset_paths.training=["'"${TRAIN_5M_DIR}"'","'"${TRAIN_7M_DIR}"'"]' \
  'inputs_train.dataset_paths.validation=["'"${VAL_5M_DIR}"'","'"${VAL_7M_DIR}"'"]' \
  inputs_train.dataset_paths.test="${TEST_5M_DIR}" \
  hydra.run.dir="${CKPT_DIR}/train_graph_rel_net"

GNN_GCC_CKPT="${CKPT_DIR}/train_gnn_gcc/logs/lightning_logs/version_0/checkpoints/last.ckpt"
GNN_SLF_CKPT="${CKPT_DIR}/train_gnn_slf/logs/lightning_logs/version_0/checkpoints/last.ckpt"
GRAPH_REL_NET_CKPT="${CKPT_DIR}/train_graph_rel_net/logs/lightning_logs/version_0/checkpoints/last.ckpt"

echo "[4/5] Evaluating SLF, GNN-GCC, GNN-SLF, GraphRelNet"
python evaluate.py --config-name experiment_full_eval \
  +inputs_eval.dataset_paths.test_4m="${TEST_4M_DIR}" \
  +inputs_eval.dataset_paths.test_5m="${TEST_5M_DIR}" \
  +inputs_eval.dataset_paths.test_6m="${TEST_6M_DIR}" \
  +inputs_eval.dataset_paths.test_7m="${TEST_7M_DIR}" \
  inputs_eval.checkpoint_paths.GNN_GCC="${GNN_GCC_CKPT}" \
  inputs_eval.checkpoint_paths.GNN_SLF="${GNN_SLF_CKPT}" \
  inputs_eval.checkpoint_paths.GraphRelNet="${GRAPH_REL_NET_CKPT}" \
  hydra.run.dir="${CKPT_DIR}/eval_all_methods"

echo "[5/5] Done"

echo "TDOA/Least Squares note: current pysoundloc submodule implementation crashes on batched grids in this repo snapshot."

popd >/dev/null
