# BatchFormer for Domain Generalization

## Installation
Code is based on Transfer-Learning-Library (https://github.com/thuml/Transfer-Learning-Library.git). One shoud first install this library, or copy the files the files in this directory into Transfer-Learning-Library/examples/domain_generalization/image_classification/

This repository has changed a few lines of code based on Transfer-Learning-Library. Check the modifications by:
```
diff baseline.py <(curl https://raw.githubusercontent.com/thuml/Transfer-Learning-Library/master/examples/domain_generalization/image_classification/baseline.py)
```


## Dataset

Following datasets can be downloaded automatically:

- [Office31](https://www.cc.gatech.edu/~judy/domainadapt/)
- [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html)
- [DomainNet](http://ai.bu.edu/M3SDA/)
- [PACS](https://domaingeneralization.github.io/#data)
- [iwildcam (WILDS)](https://wilds.stanford.edu/datasets/)
- [camelyon17 (WILDS)](https://wilds.stanford.edu/datasets/)
- [fmow (WILDS)](https://wilds.stanford.edu/datasets/)


## Experiment and Results

The shell files give the script to reproduce the [benchmarks](/docs/dglib/benchmarks/image_classification.rst) with specified hyper-parameters.
For example, if you want to reproduce IRM on Office-Home, use the following script

```shell script
# Train with IRM on Office-Home Ar Cl Rw -> Pr task using ResNet 50.
# Assume you have put the datasets under the path `data/office-home`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python irm.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50 --seed 0 --log logs/irm/OfficeHome_Pr --add_bt 1
```

Please refer to Transfer-Learning-Library for More suggestions.

if you find this repository helpful, please consider cite:

    @inproceedings{hou2022batch,
    title={BatchFormer: Learning to Explore Sample Relationships for Robust Representation Learning},
    author={Hou, Zhi and Yu, Baosheng and Tao, Dacheng},
    booktitle={CVPR},
    year={2022}
    }

