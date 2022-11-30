#!/bin/bash
#PBS -P ZHIHOU
#PBS -l select=8:ngpus=1:ncpus=6:mem=24GB
#PBS -l walltime=48:00:00
##PBS -q alloc-dt
source activate pyt

#env

cd "$PBS_O_WORKDIR"


#python run.py --nproc_per_node 1 --master_port 29543 --nnodes 8 ./configs/base.sh &>> basebf1.out

python run.py --nproc_per_node 1 --master_port 29547 --nnodes 8 ./configs/panoptic_2.sh &>> panoptic_bf3_segm.out


#python run.py --nproc_per_node 1 --master_port 29545 --nnodes 8 ./configs/panoptic.sh &>> panoptic_bf3.out

#python run.py --nproc_per_node 1 --master_port 29546 --nnodes 7 ./configs/swin.sh &>> output_bf3_7.out

