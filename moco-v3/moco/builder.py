# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class TransformerDecorator(torch.nn.Module):
    def __init__(self, add_bf=3, dim=2048, eval_global=0):
        super(TransformerDecorator, self).__init__()
        self.encoder_layers = torch.nn.TransformerEncoderLayer(dim, 4, dim, 0.5)
        self.eval_global = eval_global
        self.add_bf = add_bf

    def forward(self, feature):
        if self.training or self.eval_global > 0:
            pre_feature = feature
            feature = feature.unsqueeze(1)
            feature = self.encoder_layers(feature)
            feature = feature.squeeze(1)
            return torch.cat([pre_feature, feature], dim=0)
        return feature

"""
add_bf
32, 34, duplicate neg and pos
33, default with linear
35
"""




class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0, arch='resnet50', add_bf=0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T
        self.add_bf = add_bf

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)
        self.encoder_global=None
        if 0 < self.add_bf:
            self.encoder_global = TransformerDecorator(self.add_bf, self._get_dim(), 0)
        if self.add_bf:
            self._build_projector_and_predictor_mlps(dim, mlp_dim, self.encoder_global)
        else:
            self._build_projector_and_predictor_mlps(dim, mlp_dim)
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True, encoder_global=None):
        mlp = []
        if encoder_global:
            mlp.append(encoder_global)
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim, encoder_global=None):
        pass


    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip([k for n, k in self.base_encoder.named_parameters() if not n.__contains__('encoder_global.')],
                                    [k for n, k in self.momentum_encoder.named_parameters() if not n.__contains__('encoder_global.')],
                                    ):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)  # N*2
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)
        # the default value is 1, however, we find all choice achieves similar results.
        if self.add_bf in [1, 34, 36]:
            N = len(q1) // 2
            # self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1) 
            # this term is following the original setting: 
            # self.contrastive_loss(q1[:N], k2[:N]) + self.contrastive_loss(q2[:N], k1[:N]) # original moco loss
            # The other term is related to features with batchformer. 
            # Here, I just implement a very naive version for batchformer on Moco-V3 and report it in the paper.
            # To some extent, both batchformer and image contrastive learning investigates the sample relationships.
            # I am not familiar with contrastive learning. I guess there might be a better implementation for batchformer on moco
            loss = self.contrastive_loss(q1[:N], k2[:N]) + self.contrastive_loss(q1[:N], k2[N:]) + self.contrastive_loss(q1[N:], k2[:N]) + self.contrastive_loss(q1[N:], k2[N:]) + \
                   self.contrastive_loss(q2[:N], k1[:N]) + self.contrastive_loss(q2[:N], k1[N:]) + self.contrastive_loss(q2[N:], k1[:N]) + self.contrastive_loss(q2[N:], k1[N:])
            # emperically, all those strategies are valid and the model achieves slightly better performance when add_bf is 1.
            if self.add_bf in [1, 34]:
                return loss
            elif self.add_bf == 37:
                loss += (self.contrastive_loss(q1[:N], q1[N:].detach()) + self.contrastive_loss(q2[:N], q2[N:].detach()))
                return loss / 5.
            else:
                return loss / 4 # considering the number of losses increases
        else:
            return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim, encoder_global=None):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim, encoder_global=encoder_global)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim, encoder_global=encoder_global)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False, encoder_global=None)

    def _get_dim(self):
        return self.base_encoder.fc.weight.shape[1]

class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim, encoder_global=None):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim, encoder_global=encoder_global)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim, encoder_global=encoder_global)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)

    def _get_dim(self):
        return self.base_encoder.head.weight.shape[1]

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
