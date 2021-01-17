#!/bin/bash
export CUDA_VISIBLE_DEVICES='4,5,6,7'
NUM_PROC=$1
PORT=${PORT:-29503}
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=$PORT train.py "$@"

