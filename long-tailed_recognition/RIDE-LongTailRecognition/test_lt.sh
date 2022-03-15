#!/usr/bin/env bash

source /usr/local/anaconda/4.2.0/etc/profile.d/conda.sh

conda activate pyt
#python train_img.py -b 512 -a resnet18 data/ImageNet_LT/ --add_bt true
#
PY_ARGS=${@:1}
echo PY_ARGS
echo $PY_ARGS
#echo ${PY_ARGS}
#
python train_dist.py \
    ${PY_ARGS}
#python train_dist.py -c "configs/config_imagenet_lt_resnet50_ride_dist.json" --reduce_dimension 1 --num_experts 3 --batch_size 128 --dist-url tcp://192.168.66.220:29512 --world-size 2 --rank 0 --dist-backend nccl --multiprocessing-distributed