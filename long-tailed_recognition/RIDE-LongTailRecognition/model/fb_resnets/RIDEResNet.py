"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import autocast

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_experts, dropout=None, num_classes=1000, use_norm=False, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, share_layer3=False, returns_feat=False, s=30, add_bt=False,
                 eval_bt=0):
        self.inplanes = 64
        self.num_experts = num_experts
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.inplanes = self.next_inplanes
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.inplanes = self.next_inplanes

        self.share_layer3 = share_layer3

        if layer3_output_dim is None:
            if reduce_dimension:
                layer3_output_dim = 192
            else:
                layer3_output_dim = 256

        if layer4_output_dim is None:
            if reduce_dimension:
                layer4_output_dim = 384
            else:
                layer4_output_dim = 512

        if self.share_layer3:
            self.layer3 = self._make_layer(block, layer3_output_dim, layers[2], stride=2)
        else:
            self.layer3s = nn.ModuleList([self._make_layer(block, layer3_output_dim, layers[2], stride=2) for _ in range(num_experts)])
        self.inplanes = self.next_inplanes
        self.layer4s = nn.ModuleList([self._make_layer(block, layer4_output_dim, layers[3], stride=2) for _ in range(num_experts)])
        self.inplanes = self.next_inplanes
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.add_bt = add_bt
        self.eval_bt = eval_bt
        if add_bt:
            from model.fb_resnets import obtain_global_models
            self.encoder_layers = obtain_global_models(add_bt, self.inplanes)
            import numpy as np
            import os
            # if 'DATASET_N' in os.environ and os.environ['DATASET_N'].startswith('ImageNet'):
            #     print('load imagenet -----')
            #     cls_num = np.load('cls_num.npy')
            # else:
            #     print('load inature -----')
            #     cls_num = np.load('obj_cls_ina.npy')
            # self.many_shot_arr = np.asarray(cls_num > 100).astype(np.float32)
            # self.many_shot_arr = torch.from_numpy(self.many_shot_arr)
        self.use_dropout = True if dropout else False

        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if use_norm:
            self.linears = nn.ModuleList([NormedLinear(layer4_output_dim * block.expansion, num_classes) for _ in range(num_experts)])
        else:
            self.linears = nn.ModuleList([nn.Linear(layer4_output_dim * block.expansion, num_classes) for _ in range(num_experts)])
            s = 1
        
        self.s = s

        self.returns_feat = returns_feat

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"
        count = 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.weight.requires_grad == False:
                    module.eval()
                    count += 1

        if count > 0:
            print("Warning: detected at least one frozen BN, set them to eval state. Count:", count)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.next_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.next_inplanes, planes))

        return nn.Sequential(*layers)

    def _separate_part(self, x, ind, target=None):
        if not self.share_layer3:
            x = (self.layer3s[ind])(x)
        x = (self.layer4s[ind])(x)

        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)

        if self.use_dropout:
            x = self.dropout(x)

        if (self.training and self.add_bt > 0) or self.eval_bt:
            old_x = x
            x = x.unsqueeze(1)
            x = self.encoder_layers(x)
            x = x.squeeze(1)
            # if self.add_bt == 5:
            #     self.many_shot_arr = self.many_shot_arr.to(x.device)
            #     many_idx = self.many_shot_arr[target]
            #     many_idx = many_idx.unsqueeze(-1)
            #     # few_idx = 1 - many_idx
            #     x = old_x * many_idx + (x) * (1-many_idx)    # better for many_idx
            # elif self.add_bt == 10:
            #     self.many_shot_arr = self.many_shot_arr.to(x.device)
            #     many_idx = self.many_shot_arr[target]
            #     many_idx = many_idx.unsqueeze(-1)
            #     # few_idx = 1 - many_i
            #     x = old_x * (1 - many_idx) + (x) * many_idx   # better for few_idx
            if self.add_bt == 22:
                x = torch.cat([x, old_x], dim=0)
        self.feat.append(x)
        if self.add_bt in [32, 33] and self.training:
            # This is actually for shared classifier. and We do not use this code.
            x = (self.linears[ind])(x)
            x = x * self.s

            old_x = (self.linears[ind])(old_x)
            old_x = old_x * self.s
            return x, old_x
        x = (self.linears[ind])(x)
        x = x * self.s
        return x

    def forward(self, x, target=None):
        with autocast():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            if self.share_layer3:
                x = self.layer3(x)

            outs = []
            att_list = []
            self.feat = []
            final_out_old = None
            final_logits_old = None

            if self.add_bt in [32, 33] and self.training:
                # This is actually for shared classifier. and We do not use this code.
                outs_old = []
                for ind in range(self.num_experts):
                    out_x, out_old_x = self._separate_part(x, ind, target)
                    outs.append(out_x)
                    outs_old.append(out_old_x)
                final_out_old = torch.stack(outs_old, dim=1).mean(dim=1)
                final_logits_old = torch.stack(outs_old, dim=1)
            else:
                for ind in range(self.num_experts):
                    outs.append(self._separate_part(x, ind, target))
            final_out = torch.stack(outs, dim=1).mean(dim=1)

        if self.returns_feat:
            return {
                "output": final_out,
                "output_old": final_out_old,
                "logits_old": final_logits_old,
                "att": att_list if self.add_bt else 0,
                "feat": torch.stack(self.feat, dim=1),
                "logits": torch.stack(outs, dim=1)
            }
        else:
            return final_out
