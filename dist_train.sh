#!/usr/bin/env bash

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=2  # Adjust to the number of GPUs available

export PYTHONPATH=$(pwd):$PYTHONPATH

CUDA_VISIBLE_DEVICES=3,4 python train.py configs/spatiallm_sft_structured3d.yaml
