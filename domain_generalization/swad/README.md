# BatchFormer for SWAD

## Preparation

### Dependencies

```sh
pip install -r requirements.txt
```

### Datasets

```sh
python -m domainbed.scripts.download --data_dir=/my/datasets/path
```

### Environments

Environment details used for our study.

```
Python: 3.8.6
PyTorch: 1.7.0+cu92
Torchvision: 0.8.1+cu92
CUDA: 9.2
CUDNN: 7603
NumPy: 1.19.4
PIL: 8.0.1
```


### How to run

We provide the instructions to reproduce the main results of the paper, Table 1 and 2.
Note that the difference in a detailed environment or uncontrolled randomness may bring a little different result from the paper.

- PACS

```
python train_all.py PACS0 --dataset PACS --deterministic --trial_seed 0 --checkpoint_freq 100 --data_dir /my/datasets/path --resnet18 --add_bt 16
python train_all.py PACS1 --dataset PACS --deterministic --trial_seed 1 --checkpoint_freq 100 --data_dir /my/datasets/path --resnet18 --add_bt 16
python train_all.py PACS2 --dataset PACS --deterministic --trial_seed 2 --checkpoint_freq 100 --data_dir /my/datasets/path --resnet18 --add_bt 16
```

**You can also run the code with "--add_bt 1" which do not include shared classifier. Meanwhile, you can also set a smaller learning rate with "--lr_scale_bt 0.1".
That is also beneficial for some datasets, especially when the backbone is ResNet50.** 


## License
This code is based on SWAD.

This source code is released under the MIT license, included [here](./LICENSE).

This project includes some code from [DomainBed](https://github.com/facebookresearch/DomainBed/tree/3fe9d7bb4bc14777a42b3a9be8dd887e709ec414), also MIT licensed.
