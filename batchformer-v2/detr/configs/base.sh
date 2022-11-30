#!/usr/bin/env bash

source /usr/local/anaconda/4.2.0/etc/profile.d/conda.sh
conda activate pyt

set -x

python main.py --coco_path /project/ZHIHOU/Dataset/coco/ --bf 1 --batch_size 2 --output_dir bf1 --resume bf1/checkpoint.pth

