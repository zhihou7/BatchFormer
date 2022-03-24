## BatchFormer for Contrastive Learning 

### Introduction
Code are based on the implementation of [MoCo v3](https://arxiv.org/abs/2104.02057) for self-supervised ResNet. Code for ViT will be released after accept.

This repository has changed a few lines of code based on MoCo-v3. Check the modifications by:
```
diff main_moco.py <(curl https://raw.githubusercontent.com/facebookresearch/moco-v3/main/main_moco.py)
diff moco/builder.py <(curl https://raw.githubusercontent.com/facebookresearch/moco-v3/main/moco/builder.py)
```


### Main Results

The following results are based on ImageNet-1k self-supervised pre-training, followed by ImageNet-1k supervised training for linear evaluation or end-to-end fine-tuning. All results in these tables are based on a batch size of 4096.


### Usage: Preparation

Install PyTorch and download the ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). Similar to [MoCo v1/2](https://github.com/facebookresearch/moco), this repo contains minimal modifications on the official PyTorch ImageNet code. We assume the user can successfully run the official PyTorch ImageNet code.
For ViT models, install [timm](https://github.com/rwightman/pytorch-image-models) (`timm==0.4.9`).


### Usage: Self-supervised Pre-Training

Below are examples for MoCo v3 pre-training. 

#### ResNet-50 with 2-node (16-GPU) training, batch 4096

On the first node, run:
```
python main_moco.py \
  --moco-m-cos --crop-min=.2 \
  --dist-url 'tcp://[your first node address]:[specified port]' \
  --multiprocessing-distributed --world-size 2 --rank 0 --add_bf 1 \
  [your imagenet-folder with train and val folders]
```

bf > 0 means we use BatchFormer. 
On the second node, run the same command with `--rank 1`.
With a batch size of 4096, the training can fit into 2 nodes with a total of 16 Volta 32G GPUs. 

### Usage: Linear Classification

By default, we use momentum-SGD and a batch size of 1024 for linear classification on frozen features/weights. This can be done with a single 8-GPU node.

```
python main_lincls.py \
  -a [architecture] --lr [learning rate] \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained [your checkpoint path]/[your checkpoint file].pth.tar \
  [your imagenet-folder with train and val folders]
```

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

