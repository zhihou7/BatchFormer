#!/bin/bash
#PBS -P ZHIHOU
#PBS -l select=1:ngpus=1:ncpus=6:mem=28GB
#PBS -l walltime=24:00:00
#PBS -q alloc-xxx
source activate pyt

#env

cd "$PBS_O_WORKDIR"
# ResNet50, PACS
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s A C S -t P -a resnet50 --freeze-bn --seed 0 --log logs/baseline/PACS_P
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P C S -t A -a resnet50 --freeze-bn --seed 0 --log logs/baseline/PACS_A
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A S -t C -a resnet50 --freeze-bn --seed 0 --log logs/baseline/PACS_C
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A C -t S -a resnet50 --freeze-bn --seed 0 --log logs/baseline/PACS_S

#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s A C S -t P -a resnet50 --freeze-bn --seed 1 --log logs/baseline/PACS_P
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P C S -t A -a resnet50 --freeze-bn --seed 1 --log logs/baseline/PACS_A
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A S -t C -a resnet50 --freeze-bn --seed 1 --log logs/baseline/PACS_C
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A C -t S -a resnet50 --freeze-bn --seed 1 --log logs/baseline/PACS_S
#
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s A C S -t P -a resnet50 --freeze-bn --seed 2 --log logs/baseline/PACS_P
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P C S -t A -a resnet50 --freeze-bn --seed 2 --log logs/baseline/PACS_A
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A S -t C -a resnet50 --freeze-bn --seed 2 --log logs/baseline/PACS_C
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A C -t S -a resnet50 --freeze-bn --seed 2 --log logs/baseline/PACS_S

CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A C -t S -a resnet50 --freeze-bn --seed 0 --log logs/baseline/PACS_S_g16 --add_bt 16
CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A C -t S -a resnet50 --freeze-bn --seed 1 --log logs/baseline/PACS_S_g16 --add_bt 16
CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A C -t S -a resnet50 --freeze-bn --seed 2 --log logs/baseline/PACS_S_g16 --add_bt 16
#
#
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s A C S -t P -a resnet18 --freeze-bn --seed 0 --log logs/baseline_r18/PACS_P_g16 --add_bt 16
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P C S -t A -a resnet18 --freeze-bn --seed 0 --log logs/baseline_r18/PACS_A_g16 --add_bt 16
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A S -t C -a resnet18 --freeze-bn --seed 0 --log logs/baseline_r18/PACS_C_g16 --add_bt 16
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A C -t S -a resnet18 --freeze-bn --seed 0 --log logs/baseline_r18/PACS_S_g16 --add_bt 16
#
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s A C S -t P -a resnet18 --freeze-bn --seed 1 --log logs/baseline_r18/PACS_P_g16 --add_bt 16
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P C S -t A -a resnet18 --freeze-bn --seed 1 --log logs/baseline_r18/PACS_A_g16 --add_bt 16
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A S -t C -a resnet18 --freeze-bn --seed 1 --log logs/baseline_r18/PACS_C_g16 --add_bt 16
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A C -t S -a resnet18 --freeze-bn --seed 1 --log logs/baseline_r18/PACS_S_g16 --add_bt 16
###
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s A C S -t P -a resnet18 --freeze-bn --seed 2 --log logs/baseline_r18/PACS_P_g16 --add_bt 16
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P C S -t A -a resnet18 --freeze-bn --seed 2 --log logs/baseline_r18/PACS_A_g16 --add_bt 16
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A S -t C -a resnet18 --freeze-bn --seed 2 --log logs/baseline_r18/PACS_C_g16 --add_bt 16
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A C -t S -a resnet18 --freeze-bn --seed 2 --log logs/baseline_r18/PACS_S_g16 --add_bt 16



#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s A C S -t P -a resnet50 --freeze-bn --seed 0 --log logs/baseline/PACS_P_g16 --add_bt 16
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P C S -t A -a resnet50 --freeze-bn --seed 0 --log logs/baseline/PACS_A_g16 --add_bt 16
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A S -t C -a resnet50 --freeze-bn --seed 0 --log logs/baseline/PACS_C_g16 --add_bt 16
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A C -t S -a resnet50 --freeze-bn --seed 0 --log logs/baseline/PACS_S_g16 --add_bt 16
#
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s A C S -t P -a resnet50 --freeze-bn --seed 1 --log logs/baseline/PACS_P_g16 --add_bt 16
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P C S -t A -a resnet50 --freeze-bn --seed 1 --log logs/baseline/PACS_A_g16 --add_bt 16
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A S -t C -a resnet50 --freeze-bn --seed 1 --log logs/baseline/PACS_C_g16 --add_bt 16
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A C -t S -a resnet50 --freeze-bn --seed 1 --log logs/baseline/PACS_S_g16 --add_bt 16
#
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s A C S -t P -a resnet50 --freeze-bn --seed 2 --log logs/baseline/PACS_P_g16 --add_bt 16
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P C S -t A -a resnet50 --freeze-bn --seed 2 --log logs/baseline/PACS_A_g16 --add_bt 16
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A S -t C -a resnet50 --freeze-bn --seed 2 --log logs/baseline/PACS_C_g16 --add_bt 16
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A C -t S -a resnet50 --freeze-bn --seed 2 --log logs/baseline/PACS_S_g16 --add_bt 16

#
#
## ResNet50, Office-Home
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50 --seed 0 --log logs/baseline/OfficeHome_Pr_g16 --add_bt 16
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Ar Cl Pr -t Rw -a resnet50 --seed 0 --log logs/baseline/OfficeHome_Rw_g16 --add_bt 16
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Ar Rw Pr -t Cl -a resnet50 --seed 0 --log logs/baseline/OfficeHome_Cl_g16 --add_bt 16
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Cl Rw Pr -t Ar -a resnet50 --seed 0 --log logs/baseline/OfficeHome_Ar_g16 --add_bt 16
#
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50 --seed 1 --log logs/baseline/OfficeHome_Pr_g16 --add_bt 16;\
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Ar Cl Pr -t Rw -a resnet50 --seed 1 --log logs/baseline/OfficeHome_Rw_g16 --add_bt 16;\
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Ar Rw Pr -t Cl -a resnet50 --seed 1 --log logs/baseline/OfficeHome_Cl_g16 --add_bt 16;\
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Cl Rw Pr -t Ar -a resnet50 --seed 1 --log logs/baseline/OfficeHome_Ar_g16 --add_bt 16
#
#
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50 --seed 2 --log logs/baseline/OfficeHome_Pr_g16 --add_bt 16;\
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Ar Cl Pr -t Rw -a resnet50 --seed 2 --log logs/baseline/OfficeHome_Rw_g16 --add_bt 16;\
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Ar Rw Pr -t Cl -a resnet50 --seed 2 --log logs/baseline/OfficeHome_Cl_g16 --add_bt 16;\
#CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Cl Rw Pr -t Ar -a resnet50 --seed 2 --log logs/baseline/OfficeHome_Ar_g16 --add_bt 16



