import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from .common import MLP

class VisualProductNN(nn.Module):
    def __init__(self, dset, args):
        super(VisualProductNN, self).__init__()
        self.attr_clf = MLP(dset.feat_dim, len(dset.attrs), 2, relu = False)
        self.obj_clf = MLP(dset.feat_dim, len(dset.objs), 2, relu = False)
        self.dset = dset

    def train_forward(self, x):
        img, attrs, objs = x[0],x[1], x[2]

        attr_pred = self.attr_clf(img)
        obj_pred = self.obj_clf(img)

        attr_loss = F.cross_entropy(attr_pred, attrs)
        obj_loss = F.cross_entropy(obj_pred, objs)

        loss = attr_loss + obj_loss

        return loss, None

    def val_forward(self, x):
        img = x[0]
        attr_pred = F.softmax(self.attr_clf(img), dim =1)
        obj_pred = F.softmax(self.obj_clf(img), dim = 1)

        scores = {}
        for itr, (attr, obj) in enumerate(self.dset.pairs):
            attr_id, obj_id = self.dset.attr2idx[attr], self.dset.obj2idx[obj]
            score = attr_pred[:,attr_id] * obj_pred[:, obj_id]
            
            scores[(attr, obj)] = score
        return None, scores

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)

        return loss, pred