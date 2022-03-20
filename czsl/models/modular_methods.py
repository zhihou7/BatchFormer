# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels
import numpy as np
from .word_embedding import load_word_embeddings
import itertools
import math
import collections
from torch.distributions.bernoulli import Bernoulli
import pdb
import sys


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class GatingSampler(nn.Module):
    """Docstring for GatingSampler. """

    def __init__(self, gater, stoch_sample=True, temperature=1.0):
        """TODO: to be defined1.

        """
        nn.Module.__init__(self)
        self.gater = gater
        self._stoch_sample = stoch_sample
        self._temperature = temperature

    def disable_stochastic_sampling(self):
        self._stoch_sample = False

    def enable_stochastic_sampling(self):
        self._stoch_sample = True

    def forward(self, tdesc=None, return_additional=False, gating_wt=None):
        if self.gater is None and return_additional:
            return None, None
        elif self.gater is None:
            return None

        if gating_wt is not None:
            return_wts = gating_wt
            gating_g = self.gater(tdesc, gating_wt=gating_wt)
        else:
            gating_g = self.gater(tdesc)
            return_wts = None
            if isinstance(gating_g, tuple):
                return_wts = gating_g[1]
                gating_g = gating_g[0]

        if not self._stoch_sample:
            sampled_g = gating_g
        else:
            raise (NotImplementedError)

        if return_additional:
            return sampled_g, return_wts
        return sampled_g


class GatedModularNet(nn.Module):
    """
        An interface for creating modular nets
    """

    def __init__(self,
                 module_list,
                 start_modules=None,
                 end_modules=None,
                 single_head=False,
                 chain=False):
        """TODO: to be defined1.

        :module_list: TODO
        :g: TODO

        """
        nn.Module.__init__(self)
        self._module_list = nn.ModuleList(
            [nn.ModuleList(m) for m in module_list])

        self.num_layers = len(self._module_list)
        if start_modules is not None:
            self._start_modules = nn.ModuleList(start_modules)
        else:
            self._start_modules = None

        if end_modules is not None:
            self._end_modules = nn.ModuleList(end_modules)
            self.num_layers += 1
        else:
            self._end_modules = None

        self.sampled_g = None
        self.single_head = single_head
        self._chain = chain

    def forward(self, x, sampled_g=None, t=None, return_feat=False):
        """TODO: Docstring for forward.

        :x: Input data
        :g: Gating tensor (#Task x )#num_layer x #num_mods x #num_mods
        :t: task ID
        :returns: TODO

        """

        if t is None:
            t_not_set = True
            t = torch.tensor([0] * x.shape[0], dtype=x.dtype).long()
        else:
            t_not_set = False
            t = t.squeeze()

        if self._start_modules is not None:
            prev_out = [mod(x) for mod in self._start_modules]
        else:
            prev_out = [x]

        if sampled_g is None:
            # NON-Gated Module network
            prev_out = sum(prev_out) / float(len(prev_out))
            #prev_out = torch.mean(prev_out, 0)
            for li in range(len(self._module_list)):
                prev_out = sum([
                    mod(prev_out) for mod in self._module_list[li]
                ]) / float(len(self._module_list[li]))
            features = prev_out
            if self._end_modules is not None:
                if t_not_set or self.single_head:
                    prev_out = self._end_modules[0](prev_out)
                else:
                    prev_out = torch.cat([
                        self._end_modules[tid](prev_out[bi:bi + 1])
                        for bi, tid in enumerate(t)
                    ], 0)
            if return_feat:
                return prev_out, features
            return prev_out
        else:
            # Forward prop with sampled Gs
            for li in range(len(self._module_list)):
                curr_out = []
                for j in range(len(self._module_list[li])):
                    gind = j if not self._chain else 0
                    # Dim: #Batch x C
                    module_in_wt = sampled_g[li + 1][gind]
                    # Module input weights rearranged to match inputs
                    module_in_wt = module_in_wt.transpose(0, 1)
                    add_dims = prev_out[0].dim() + 1 - module_in_wt.dim()
                    module_in_wt = module_in_wt.view(*module_in_wt.shape,
                                                     *([1] * add_dims))
                    module_in_wt = module_in_wt.expand(
                        len(prev_out), *prev_out[0].shape)
                    module_in = sum([
                        module_in_wt[i] * prev_out[i]
                        for i in range(len(prev_out))
                    ])
                    mod = self._module_list[li][j]
                    curr_out.append(mod(module_in))
                prev_out = curr_out

            # Output modules (with sampled Gs)
            if self._end_modules is not None:
                li = self.num_layers - 1
                if t_not_set or self.single_head:
                    # Dim: #Batch x C
                    module_in_wt = sampled_g[li + 1][0]
                    # Module input weights rearranged to match inputs
                    module_in_wt = module_in_wt.transpose(0, 1)
                    add_dims = prev_out[0].dim() + 1 - module_in_wt.dim()
                    module_in_wt = module_in_wt.view(*module_in_wt.shape,
                                                     *([1] * add_dims))
                    module_in_wt = module_in_wt.expand(
                        len(prev_out), *prev_out[0].shape)
                    module_in = sum([
                        module_in_wt[i] * prev_out[i]
                        for i in range(len(prev_out))
                    ])
                    features = module_in
                    prev_out = self._end_modules[0](module_in)
                else:
                    curr_out = []
                    for bi, tid in enumerate(t):
                        # Dim: #Batch x C
                        gind = tid if not self._chain else 0
                        module_in_wt = sampled_g[li + 1][gind]
                        # Module input weights rearranged to match inputs
                        module_in_wt = module_in_wt.transpose(0, 1)
                        add_dims = prev_out[0].dim() + 1 - module_in_wt.dim()
                        module_in_wt = module_in_wt.view(
                            *module_in_wt.shape, *([1] * add_dims))
                        module_in_wt = module_in_wt.expand(
                            len(prev_out), *prev_out[0].shape)
                        module_in = sum([
                            module_in_wt[i] * prev_out[i]
                            for i in range(len(prev_out))
                        ])
                        features = module_in
                        mod = self._end_modules[tid]
                        curr_out.append(mod(module_in[bi:bi + 1]))
                    prev_out = curr_out
                    prev_out = torch.cat(prev_out, 0)
            if return_feat:
                return prev_out, features
            return prev_out


class CompositionalModel(nn.Module):
    def __init__(self, dset, args):
        super(CompositionalModel, self).__init__()
        self.args = args
        self.dset = dset

        # precompute validation pairs
        attrs, objs = zip(*self.dset.pairs)
        attrs = [dset.attr2idx[attr] for attr in attrs]
        objs = [dset.obj2idx[obj] for obj in objs]
        self.val_attrs = torch.LongTensor(attrs).cuda()
        self.val_objs = torch.LongTensor(objs).cuda()

    def train_forward_softmax(self, x):
        img, attrs, objs = x[0], x[1], x[2]
        neg_attrs, neg_objs = x[4], x[5]
        inv_attrs, comm_attrs = x[6], x[7]

        sampled_attrs = torch.cat((attrs.unsqueeze(1), neg_attrs), 1)
        sampled_objs = torch.cat((objs.unsqueeze(1), neg_objs), 1)
        img_ind = torch.arange(sampled_objs.shape[0]).unsqueeze(1).repeat(
            1, sampled_attrs.shape[1])

        flat_sampled_attrs = sampled_attrs.view(-1)
        flat_sampled_objs = sampled_objs.view(-1)
        flat_img_ind = img_ind.view(-1)
        labels = torch.zeros_like(sampled_attrs[:, 0]).long()

        self.composed_g = self.compose(flat_sampled_attrs, flat_sampled_objs)

        cls_scores, feat = self.comp_network(
            img[flat_img_ind], self.composed_g, return_feat=True)
        pair_scores = cls_scores[:, :1]
        pair_scores = pair_scores.view(*sampled_attrs.shape)

        loss = 0
        loss_cls = F.cross_entropy(pair_scores, labels)
        loss += loss_cls

        loss_obj = torch.FloatTensor([0])
        loss_attr = torch.FloatTensor([0])
        loss_sparse = torch.FloatTensor([0])
        loss_unif = torch.FloatTensor([0])
        loss_aux = torch.FloatTensor([0])

        acc = (pair_scores.argmax(1) == labels).sum().float() / float(
            len(labels))
        all_losses = {}
        all_losses['total_loss'] = loss
        all_losses['main_loss'] = loss_cls
        all_losses['aux_loss'] = loss_aux
        all_losses['obj_loss'] = loss_obj
        all_losses['attr_loss'] = loss_attr
        all_losses['sparse_loss'] = loss_sparse
        all_losses['unif_loss'] = loss_unif

        return loss, all_losses, acc, (pair_scores, feat)

    def val_forward(self, x):
        img = x[0]
        batch_size = img.shape[0]
        pair_scores = torch.zeros(batch_size, len(self.val_attrs))
        pair_feats = torch.zeros(batch_size, len(self.val_attrs),
                                 self.args.emb_dim)
        pair_bs = len(self.val_attrs)

        for pi in range(math.ceil(len(self.val_attrs) / pair_bs)):
            self.compose_g = self.compose(
                self.val_attrs[pi * pair_bs:(pi + 1) * pair_bs],
                self.val_objs[pi * pair_bs:(pi + 1) * pair_bs])
            compose_g = self.compose_g
            expanded_im = img.unsqueeze(1).repeat(
                1, compose_g[0][0].shape[0],
                *tuple([1] * (img.dim() - 1))).view(-1, *img.shape[1:])
            expanded_compose_g = [[
                g.unsqueeze(0).repeat(batch_size, *tuple([1] * g.dim())).view(
                    -1, *g.shape[1:]) for g in layer_g
            ] for layer_g in compose_g]
            this_pair_scores, this_feat = self.comp_network(
                expanded_im, expanded_compose_g, return_feat=True)
            featnorm = torch.norm(this_feat, p=2, dim=-1)
            this_feat = this_feat.div(
                featnorm.unsqueeze(-1).expand_as(this_feat))

            this_pair_scores = this_pair_scores[:, :1].view(batch_size, -1)
            this_feat = this_feat.view(batch_size, -1, self.args.emb_dim)

            pair_scores[:, pi * pair_bs:pi * pair_bs +
                        this_pair_scores.shape[1]] = this_pair_scores[:, :]
            pair_feats[:, pi * pair_bs:pi * pair_bs +
                       this_pair_scores.shape[1], :] = this_feat[:]

        scores = {}
        feats = {}
        for i, (attr, obj) in enumerate(self.dset.pairs):
            scores[(attr, obj)] = pair_scores[:, i]
            feats[(attr, obj)] = pair_feats[:, i]

        # return None, (scores, feats)
        return None, scores

    def forward(self, x, with_grad=False):
        if self.training:
            loss, loss_aux, acc, pred = self.train_forward(x)
        else:
            loss_aux = torch.Tensor([0])
            loss = torch.Tensor([0])
            if not with_grad:
                with torch.no_grad():
                    acc, pred = self.val_forward(x)
            else:
                acc, pred = self.val_forward(x)
        # return loss, loss_aux, acc, pred
        return loss, pred


class GatedGeneralNN(CompositionalModel):
    """Docstring for GatedCompositionalModel. """

    def __init__(self,
                 dset,
                 args,
                 num_layers=2,
                 num_modules_per_layer=3,
                 stoch_sample=False,
                 use_full_model=False,
                 num_classes=[2],
                 gater_type='general'):
        """TODO: to be defined1.

        :dset: TODO
        :args: TODO

        """
        CompositionalModel.__init__(self, dset, args)

        self.train_forward = self.train_forward_softmax
        self.compose_type = 'nn' #todo: could be different

        gating_in_dim = 128
        if args.emb_init:
            gating_in_dim = 300
        elif args.clf_init:
            gating_in_dim = 512

        if self.compose_type == 'nn':
            tdim = gating_in_dim * 2
            inter_tdim = self.args.embed_rank
            # Change this to allow only obj, only attr gatings
            self.attr_embedder = nn.Embedding(
                len(dset.attrs) + 1,
                gating_in_dim,
                padding_idx=len(dset.attrs),
            )
            self.obj_embedder = nn.Embedding(
                len(dset.objs) + 1,
                gating_in_dim,
                padding_idx=len(dset.objs),
            )

            # initialize the weights of the embedders with the svm weights
            if args.emb_init:
                pretrained_weight = load_word_embeddings(
                    args.emb_init, dset.attrs)
                self.attr_embedder.weight[:-1, :].data.copy_(pretrained_weight)
                pretrained_weight = load_word_embeddings(
                    args.emb_init, dset.objs)
                self.obj_embedder.weight.data[:-1, :].copy_(pretrained_weight)
            elif args.clf_init:
                for idx, attr in enumerate(dset.attrs):
                    at_id = self.dset.attr2idx[attr]
                    weight = torch.load(
                        '%s/svm/attr_%d' % (args.data_dir,
                                            at_id)).coef_.squeeze()
                    self.attr_embedder.weight[idx].data.copy_(
                        torch.from_numpy(weight))
                for idx, obj in enumerate(dset.objs):
                    obj_id = self.dset.obj2idx[obj]
                    weight = torch.load(
                        '%s/svm/obj_%d' % (args.data_dir,
                                           obj_id)).coef_.squeeze()
                    self.obj_embedder.weight[idx].data.copy_(
                        torch.from_numpy(weight))
            else:
                n_attr = len(dset.attrs)
                gating_in_dim = 300
                tdim = gating_in_dim * 2 + n_attr
                self.attr_embedder = nn.Embedding(
                    n_attr,
                    n_attr,
                )
                self.attr_embedder.weight.data.copy_(
                    torch.from_numpy(np.eye(n_attr)))
                self.obj_embedder = nn.Embedding(
                    len(dset.objs) + 1,
                    gating_in_dim,
                    padding_idx=len(dset.objs),
                )
                pretrained_weight = load_word_embeddings(
                    '/home/ubuntu/workspace/czsl/data/glove/glove.6B.300d.txt', dset.objs)
                self.obj_embedder.weight.data[:-1, :].copy_(pretrained_weight)
        else:
            raise (NotImplementedError)

        self.comp_network, self.gating_network, self.nummods, _ = modular_general(
            num_layers=num_layers,
            num_modules_per_layer=num_modules_per_layer,
            feat_dim=dset.feat_dim,
            inter_dim=args.emb_dim,
            stoch_sample=stoch_sample,
            use_full_model=use_full_model,
            tdim=tdim,
            inter_tdim=inter_tdim,
            gater_type=gater_type,
        )

        if args.static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

    def compose(self, attrs, objs):
        obj_wt = self.obj_embedder(objs)
        if self.compose_type == 'nn':
            attr_wt = self.attr_embedder(attrs)
            inp_wts = torch.cat([attr_wt, obj_wt], 1)  # 2D
        else:
            raise (NotImplementedError)
        composed_g, composed_g_wt = self.gating_network(
            inp_wts, return_additional=True)

        return composed_g


class GeneralNormalizedNN(nn.Module):
    """Docstring for GatedCompositionalModel. """

    def __init__(self, num_layers, num_modules_per_layer, in_dim, inter_dim):
        """TODO: to be defined1. """
        nn.Module.__init__(self)
        self.start_modules = [nn.Sequential()]
        self.layer1 = [[
            nn.Sequential(
                nn.Linear(in_dim, inter_dim), nn.BatchNorm1d(inter_dim),
                nn.ReLU()) for _ in range(num_modules_per_layer)
        ]]
        if num_layers > 1:
            self.layer2 = [[
                nn.Sequential(
                    nn.BatchNorm1d(inter_dim), nn.Linear(inter_dim, inter_dim),
                    nn.BatchNorm1d(inter_dim), nn.ReLU())
                for _m in range(num_modules_per_layer)
            ] for _l in range(num_layers - 1)]
        self.avgpool = nn.Sequential()
        self.fc = [
            nn.Sequential(nn.BatchNorm1d(inter_dim), nn.Linear(inter_dim, 1))
        ]


class GeneralGatingNN(nn.Module):
    def __init__(
            self,
            num_mods,
            tdim,
            inter_tdim,
            randinit=False,
    ):
        """TODO: to be defined1.

        :num_mods: TODO
        :tdim: TODO

        """
        nn.Module.__init__(self)

        self._num_mods = num_mods
        self._tdim = tdim
        self._inter_tdim = inter_tdim
        task_outdim = self._inter_tdim

        self.task_linear1 = nn.Linear(self._tdim, task_outdim, bias=False)
        self.task_bn1 = nn.BatchNorm1d(task_outdim)
        self.task_linear2 = nn.Linear(task_outdim, task_outdim, bias=False)
        self.task_bn2 = nn.BatchNorm1d(task_outdim)
        self.joint_linear1 = nn.Linear(task_outdim, task_outdim, bias=False)
        self.joint_bn1 = nn.BatchNorm1d(task_outdim)

        num_out = [[1]] + [[
            self._num_mods[i - 1] for _ in range(self._num_mods[i])
        ] for i in range(1, len(self._num_mods))]
        count = 0
        out_ind = []
        for i in range(len(num_out)):
            this_out_ind = []
            for j in range(len(num_out[i])):
                this_out_ind.append([count, count + num_out[i][j]])
                count += num_out[i][j]
            out_ind.append(this_out_ind)
        self.out_ind = out_ind
        self.out_count = count

        self.joint_linear2 = nn.Linear(task_outdim, count, bias=False)

        def apply_init(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.ModuleList):
                for subm in m:
                    if isinstance(subm, nn.ModuleList):
                        for subsubm in subm:
                            apply_init(subsubm)
                    else:
                        apply_init(subm)
            else:
                apply_init(m)

        if not randinit:
            self.joint_linear2.weight.data.zero_()

    def forward(self, tdesc=None):
        """TODO: Docstring for function.

        :arg1: TODO
        :returns: TODO

        """
        if tdesc is None:
            return None

        x = tdesc
        task_embeds1 = F.relu(self.task_bn1(self.task_linear1(x)))
        joint_embed = task_embeds1
        joint_embed = self.joint_linear2(joint_embed)
        joint_embed = [[
            joint_embed[:, self.out_ind[i][j][0]:self.out_ind[i][j][1]]
            for j in range(len(self.out_ind[i]))
        ] for i in range(len(self.out_ind))]

        gating_wt = joint_embed
        prob_g = [[F.softmax(wt, -1) for wt in gating_wt[i]]
                  for i in range(len(gating_wt))]
        return prob_g, gating_wt


def modularize_network(
        model,
        stoch_sample=False,
        use_full_model=False,
        tdim=200,
        inter_tdim=200,
        gater_type='general',
        single_head=True,
        num_classes=[2],
        num_lookup_gating=10,
):
    # Copy start modules and end modules
    start_modules = model.start_modules
    end_modules = [
        nn.Sequential(model.avgpool, Flatten(), fci) for fci in model.fc
    ]

    # Create module_list as list of lists [[layer1 modules], [layer2 modules], ...]
    module_list = []
    li = 1
    while True:
        if hasattr(model, 'layer{}'.format(li)):
            module_list.extend(getattr(model, 'layer{}'.format(li)))
            li += 1
        else:
            break

    num_module_list = [len(start_modules)] + [len(layer) for layer in module_list] \
            + [len(end_modules)]
    gated_model_func = GatedModularNet
    gated_net = gated_model_func(
        module_list,
        start_modules=start_modules,
        end_modules=end_modules,
        single_head=single_head)

    gater_func = GeneralGatingNN
    gater = gater_func(
        num_mods=num_module_list, tdim=tdim, inter_tdim=inter_tdim)

    fan_in = num_module_list
    if use_full_model:
        gater = None

    # Create Gating Sampler
    gating_sampler = GatingSampler(gater=gater, stoch_sample=stoch_sample)
    return gated_net, gating_sampler, num_module_list, fan_in


def modular_general(
        num_layers,
        num_modules_per_layer,
        feat_dim,
        inter_dim,
        stoch_sample=False,
        use_full_model=False,
        num_classes=[2],
        single_head=True,
        gater_type='general',
        tdim=300,
        inter_tdim=300,
        num_lookup_gating=10,
):
    # First create a ResNext model
    model = GeneralNormalizedNN(
        num_layers,
        num_modules_per_layer,
        feat_dim,
        inter_dim,
    )

    # Modularize the model and create gating funcs
    gated_net, gating_sampler, num_module_list, fan_in = modularize_network(
        model,
        stoch_sample,
        use_full_model=use_full_model,
        gater_type=gater_type,
        tdim=tdim,
        inter_tdim=inter_tdim,
        single_head=single_head,
        num_classes=num_classes,
        num_lookup_gating=num_lookup_gating,
    )
    return gated_net, gating_sampler, num_module_list, fan_in