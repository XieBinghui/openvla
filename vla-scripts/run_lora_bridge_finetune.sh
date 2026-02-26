#!/usr/bin/env bash
set -euo pipefail

# 只需要改这个路径（里面应包含 bridge_orig/）
DATA_ROOT_DIR="/mnt/kaiwu-group-x6/binghuixie/bridge_orig"
RUN_ROOT_DIR="/root/code/openvla/checkpoints"
ADAPTER_TMP_DIR="./adapter-tmp"
LOG_DIR="./logs"

# 无网环境：使用离线 W&B（仍可记录训练曲线）
export WANDB_MODE=offline
export WANDB_DIR="${LOG_DIR}/wandb"

mkdir -p "${RUN_ROOT_DIR}" "${ADAPTER_TMP_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/lora_bridge_$(date +%Y%m%d_%H%M%S).log"

torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir "${DATA_ROOT_DIR}" \
  --dataset_name bridge_orig \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --adapter_tmp_dir "${ADAPTER_TMP_DIR}" \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 7e-4 \
  --image_aug True \
  --save_steps 2000 \
  --wandb_project openvla \
  --wandb_entity offline \
  2>&1 | tee "${LOG_FILE}"
