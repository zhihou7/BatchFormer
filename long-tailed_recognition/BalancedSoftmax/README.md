# BatchFormer for BalancedSoftmax

Code for the paper BatchFormer on BalancedSoftmax. This repository includes the ablation study codes and thus might also include some redudanct code.

## Requirements 
* Python 3
* [PyTorch](https://pytorch.org/) (version == 1.4)
* [yaml](https://pyyaml.org/wiki/PyYAMLDocumentation)
* [higher](https://github.com/facebookresearch/higher)(version == 0.2.1)



## Training
### End-to-end Training

- Balanced Softmax
```bash
python main.py --cfg ./config/ImageNet_LT/balanced_softmax_resnet10.yaml
```

- BatchFormer
```bash
python main.py --cfg ./config/ImageNet_LT/balanced_softmax_resnet10.yaml --add_bt 1
```

## Evaluation

Model evaluation can be done using the following command:
```bash
python main.py --cfg ./config/CIFAR10_LT/balanced_softmax_imba200.yaml --test
```


## Citation
```bibtex
@inproceedings{
    Ren2020balms,
    title={Balanced Meta-Softmax for Long-Tailed Visual Recognition},
    author={Jiawei Ren and Cunjun Yu and Shunan Sheng and Xiao Ma and Haiyu Zhao and Shuai Yi and Hongsheng Li},
    booktitle={Proceedings of Neural Information Processing Systems(NeurIPS)},
    month = {Dec},
    year={2020}
}
```


```bibtex
@inproceedings{
    Ren2020balms,
    title={Balanced Meta-Softmax for Long-Tailed Visual Recognition},
    author={Jiawei Ren and Cunjun Yu and Shunan Sheng and Xiao Ma and Haiyu Zhao and Shuai Yi and Hongsheng Li},
    booktitle={Proceedings of Neural Information Processing Systems(NeurIPS)},
    month = {Dec},
    year={2020}
}
```

Code are based on [Balanced Meta-Softmax](https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification.git)
