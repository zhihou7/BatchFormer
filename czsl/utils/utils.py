import os
from os.path import join as ospj
import torch
import random
import copy
import shutil
import sys
import yaml


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_norm_values(norm_family = 'imagenet'):
    '''
        Inputs
            norm_family: String of norm_family
        Returns
            mean, std : tuple of 3 channel values
    '''
    if norm_family == 'imagenet':
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        raise ValueError('Incorrect normalization family')
    return mean, std

def save_args(args, log_path, argfile):
    shutil.copy('train.py', log_path)
    modelfiles = ospj(log_path, 'models')
    try:
        shutil.copy(argfile, log_path)
    except:
        print('Config exists')
    try:
        shutil.copytree('models/', modelfiles)
    except:
        print('Already exists')
    with open(ospj(log_path,'args_all.yaml'),'w') as f:
        yaml.dump(args, f, default_flow_style=False, allow_unicode=True)
    with open(ospj(log_path, 'args.txt'), 'w') as f:
        f.write('\n'.join(sys.argv[1:]))

class UnNormalizer:
    '''
    Unnormalize a given tensor using mean and std of a dataset family
    Inputs
        norm_family: String, dataset
        tensor: Torch tensor
    Outputs
        tensor: Unnormalized tensor
    '''
    def __init__(self, norm_family = 'imagenet'):
        self.mean, self.std = get_norm_values(norm_family=norm_family)
        self.mean, self.std = torch.Tensor(self.mean).view(1, 3, 1, 1), torch.Tensor(self.std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        return (tensor * self.std) + self.mean

def load_args(filename, args):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for key, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)