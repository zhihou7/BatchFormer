# BatchFormer for Domain Generalization

## Installation
Code is based on Transfer-Learning-Library (https://github.com/thuml/Transfer-Learning-Library.git). One shoud first install this library, or copy the files the files in this directory into Transfer-Learning-Library/examples/domain_generalization/image_classification/



This repository has changed a few lines of code based on Transfer-Learning-Library. Check the modifications by:
```
diff baseline.py <(curl https://raw.githubusercontent.com/thuml/Transfer-Learning-Library/master/examples/domain_generalization/image_classification/baseline.py)
```

I have also included Transfer-Learning-Library in this directory. You can find the examples in Transfer-Learning-Library/master/examples/domain_generalization/image_classification/ (e.g. baseline.py, irm.py)

In Transfer-Learning-Library/master/examples/domain_adaption/image_classification/, you can find corresponding code for Domain Adaption.


BatchFormer based on SWAD is provided in swad.


## Experiment and Results

For example, if you want to reproduce IRM on Office-Home, use the following script

```shell script
CUDA_VISIBLE_DEVICES=0 python irm.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50 --seed 0 --log logs/irm/OfficeHome_Pr --add_bt 1
```

You can find the corresponding *.pbs files in Transfer-Learning-Library/master/examples/domain_generalization/image_classification/ to get the running scripts for other methods. Please refer to Transfer-Learning-Library for More suggestions.

if you find this repository helpful, please consider cite:

    @inproceedings{hou2022batch,
    title={BatchFormer: Learning to Explore Sample Relationships for Robust Representation Learning},
    author={Hou, Zhi and Yu, Baosheng and Tao, Dacheng},
    booktitle={CVPR},
    year={2022}
    }

