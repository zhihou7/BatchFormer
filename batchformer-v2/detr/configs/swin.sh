#!/usr/bin/env bash

source /usr/local/anaconda/4.2.0/etc/profile.d/conda.sh
conda activate pyt

set -x

python main.py --coco_path /project/ZHIHOU/Dataset/coco/ --bf 3 --batch_size 2 --output_dir output_bf3_7 --resume output_bf3_7/checkpoint.pth

