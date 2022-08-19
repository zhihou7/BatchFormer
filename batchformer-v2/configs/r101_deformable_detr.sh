#!/usr/bin/env bash

source /usr/local/anaconda/4.2.0/etc/profile.d/conda.sh
conda activate pyt

set -x

EXP_DIR=exps/r101_deformable_detr
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
