
# BatchFormer for Compositional Zero-Shot Learning


## Setup 

1. Clone the repo and 

```
cd czsl
```

2. We recommend using Anaconda for environment setup. To create the environment and activate it, please run:
```
    conda env create --file environment.yml
    conda activate czsl
```

4. Go to the cloned repo and open a terminal. Download the datasets and embeddings, specifying the desired path (e.g. `DATA_ROOT` in the example):
```
    bash ./utils/download_data.sh DATA_ROOT
    mkdir logs
```

## Training
**Closed World.** To train a model, the command is simply:
```
    python train.py --config CONFIG_FILE
```
where `CONFIG_FILE` is the path to the configuration file of the model. 
The folder `configs` contains configuration files for all methods, i.e. CGE in `configs/cge`, CompCos in `configs/compcos`, and the other methods in `configs/baselines`.  

To run CGE on MitStates, the command is just:
```
    python train.py --config configs/cge/mit.yml --add_bt 1
```
On UT-Zappos, the command is:
```
    python train.py --config configs/cge/utzappos.yml --add_bt 1
```

## Test
 

**Closed World.** To test a model, the code is simple:
```
    python test.py --logpath LOG_DIR
```
where `LOG_DIR` is the directory containing the logs of a model.


## References
If you use this code, please cite
```
@inproceedings{hou2022batch,
    title={BatchFormer: Learning to Explore Sample Relationships for Robust Representation Learning},
    author={Hou, Zhi and Yu, Baosheng and Tao, Dacheng},
    booktitle={CVPR},
    year={2022}
}
```

**Note**: Code are based on czsl:
```
@inproceedings{naeem2021learning,
  title={Learning Graph Embeddings for Compositional Zero-shot Learning},
  author={Naeem, MF and Xian, Y and Tombari, F and Akata, Zeynep},
  booktitle={34th IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021},
  organization={IEEE}
}
```
