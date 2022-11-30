#!/usr/bin/env bash

source /usr/local/anaconda/4.2.0/etc/profile.d/conda.sh
conda activate pyt

set -x

#python main.py --coco_path /project/ZHIHOU/Dataset/coco/ --bf 1 --batch_size 2 --output_dir bf1

#python -m torch.distributed.launch --nproc_per_node=8 --use_env
python main.py --coco_path /project/ZHIHOU/Dataset/coco/  \
--coco_panoptic_path /project/ZHIHOU/Dataset/coco/ \
--dataset_file coco_panoptic \
--output_dir panoptic_bf3 --bf 3 --resume panoptic_bf3/checkpoint.pth