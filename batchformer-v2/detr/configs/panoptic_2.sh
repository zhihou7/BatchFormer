#!/usr/bin/env bash

source /usr/local/anaconda/4.2.0/etc/profile.d/conda.sh
conda activate pyt

set -x


python main.py --masks --epochs 25 --lr_drop 15 --coco_path /project/ZHIHOU/Dataset/coco/  \
--coco_panoptic_path /project/ZHIHOU/Dataset/coco/  \
--dataset_file coco_panoptic \
--frozen_weights panoptic_bf3/checkpoint0249.pth \
--use_checkpoint \
--output_dir panoptic_bf3_segm --bf 3 --batch_size 2