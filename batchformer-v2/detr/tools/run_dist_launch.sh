#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
#source ~/.bashrc
#conda activate torch150

source /usr/local/anaconda/4.2.0/etc/profile.d/conda.sh
conda activate pyt
#module load gcc
set -x

GPUS=$1
RUN_COMMAND=${@:2}
#if [ $GPUS -lt 8 ]; then
#    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
#else
#    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
#fi
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29500"}
NODE_RANK=${NODE_RANK:-0}
BASE_RANK=${BASE_RANK:-0}
#env
echo $GPUS
echo $GPUS_PER_NODE

#let "NNODES=GPUS/GPUS_PER_NODE"

python ./tools/launch.py \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --base_rank ${BASE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --nproc_per_node ${GPUS_PER_NODE} \
    ${RUN_COMMAND}