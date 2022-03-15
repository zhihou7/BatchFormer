# BatchFormer for Long-Tailed Recognition (RIDE)

Different from other methods that BatchFormer is applied for, it is not required to add a shared classifier for RIDE. 
In this repository, we provide the version without shared classifier.

## Requirements
### Packages
* Python >= 3.7, < 3.9
* PyTorch >= 1.6
* tqdm (Used in `test.py`)
* tensorboard >= 1.14 (for visualization)
* pandas
* numpy

### Hardware requirements
We evaluate BatchFormer with 3 experts RIDE on 4 V100(16G) GPUs.


## Dataset Preparation
CIFAR code will download data automatically with the dataloader. We use data the same way as [classifier-balancing](https://github.com/facebookresearch/classifier-balancing). For ImageNet-LT and iNaturalist, please prepare data in the `data` directory. ImageNet-LT can be found at [this link](https://drive.google.com/drive/u/1/folders/1j7Nkfe6ZhzKFXePHdsseeeGI877Xu1yf). iNaturalist data should be the 2018 version from [this](https://github.com/visipedia/inat_comp) repo (Note that it requires you to pay to download now). The annotation can be found at [here](https://github.com/facebookresearch/classifier-balancing/tree/master/data). Please put them in the same location as below:

```
data
├── cifar-100-python
│   ├── file.txt~
│   ├── meta
│   ├── test
│   └── train
├── cifar-100-python.tar.gz
├── ImageNet_LT
│   ├── ImageNet_LT_open.txt
│   ├── ImageNet_LT_test.txt
│   ├── ImageNet_LT_train.txt
│   ├── ImageNet_LT_val.txt
│   ├── test
│   ├── train
│   └── val
└── iNaturalist18
    ├── iNaturalist18_train.txt
    ├── iNaturalist18_val.txt
    └── train_val2018
```


## Training and Evaluation Instructions

### ImageNet-LT
#### ResNet 10

```
python train.py -c "configs/config_imagenet_lt_resnet10_ride.json" --reduce_dimension 1 --num_experts 3 --add_bt 1
```

##### ResNet 50
```
python train.py -c "configs/config_imagenet_lt_resnet50_ride.json" --reduce_dimension 1 --num_experts 3 --add_bt 1 --batch_size 400
```

or,
```
python run_imagenet.py --nproc_per_node 1 --master_port 29515 --nnodes 4 ./test_lt.sh -c "configs/config_imagenet_lt_resnet50_ride_dist.json" \
--reduce_dimension 1 --num_experts 3 --batch_size 100 --add_global 1
```
Due to the limitation of GPUs, we usually use 4 GPUs from different machines (i.e., distribution training). 
Due to limitation of copyright, we do not provide the file of run_imagenet.py, which actually run the commands (single GPU) in each node.  
In our experiment, if we suffer from OOM, we usually reduce the batch size. For example, we might reduce 128 to 120 or 100. 
We make sure the baseline is also trained with the same settings.

##### ResNeXt 50
```
python train.py -c "configs/config_imagenet_lt_resnext50_ride.json" --reduce_dimension 1 --num_experts 3 --add_bt 1
```

### iNaturalist
```
python train.py -c "configs/config_iNaturalist_resnet50_ride.json" --reduce_dimension 1 --num_experts 3 --add_bt 1
```

### Test
To test a checkpoint, please put it with the corresponding config file.
```
python test.py -r path_to_checkpoint
```

Please see [the pytorch template that we use](https://github.com/victoresque/pytorch-template) for additional more general usages of this project (e.g. loading from a checkpoint, etc.).

## Citation
If you find our work inspiring or use our codebase in your research, please cite our work.
```
@inproceedings{hou2022batch,
    title={BatchFormer: Learning to Explore Sample Relationships for Robust Representation Learning},
    author={Hou, Zhi and Yu, Baosheng and Tao, Dacheng},
    booktitle={CVPR},
    year={2022}
}
```

### License
This project is licensed under the MIT License. See [LICENSE](https://github.com/frank-xwang/RIDE-LongTailRecognition/blob/main/LICENSE) for more details. The parts described below follow their original license.

## Acknowledgements
This is a project based on this [RIDE](https://github.com/frank-xwang/RIDE-LongTailRecognition.git).