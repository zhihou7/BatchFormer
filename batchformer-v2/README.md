# BatchFormerV2: Exploring Sample Relationships for Dense Representation Learning

This repository is based on Deformable DETR. One can follow the instructions in Deformable DETR to install Deformable DETR first.

This repository has changed a few lines of the core code based on Deformable DETR. Check the modifications by:
```
diff models/deformable_transformer.py <(curl https://raw.githubusercontent.com/fundamentalvision/Deformable-DETR/main/models/deformable_transformer.py)
```

## Main Results

| Method      | AP |
| ----------- | ----------- |
| Deformable DETR  | 43.8       |
| + BatchFormerV2   | 45.5        |

## Installation

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

## Usage

### Dataset preparation

Please download [COCO 2017 dataset](https://cocodataset.org/) and organize them as following:

```
code_root/
└── data/
    └── coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
        	├── instances_train2017.json
        	└── instances_val2017.json
```

### Training

#### Training on single node

For example, the command for training Deformable DETR on 8 GPUs is as following:

```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/r50_deformable_detr.sh --batch-size 3 --share_bf 0 --bf 1 --insert_idx 0
```

"--share_bf 0 --bf 1 --insert_idx 0" is parameters for BatchFormerV2

### Evaluation

You can get the config file and pretrained model of Deformable DETR (the link is in "Main Results" session), then run following command to evaluate it on COCO 2017 validation set:

```bash
<path to config file> --resume <path to pre-trained model> --eval
```

If you find this repository is helpful, please consider to cite


    @inproceedings{hou2022batch,
    	title={BatchFormer: Learning to Explore Sample Relationships for Robust Representation Learning},
	author={Hou, Zhi and Yu, Baosheng and Tao, Dacheng},
    	booktitle={CVPR},
	year={2022}
	}
      @article{hou2022batchformerv2,
 	 title={BatchFormerV2: Exploring Sample Relationships for Dense Representation Learning},
	  author={Hou, Zhi and Yu, Baosheng and Wang, Chaoyue and Zhan, Yibing and Tao, Dacheng},
	  journal={arXiv preprint arXiv:2204.01254},
	  year={2022}
	}
	
