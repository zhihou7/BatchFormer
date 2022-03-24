import torch
import torch.nn as nn


class TransformerDecorator(torch.nn.Module):
    def __init__(self, add_bf=0, dim=2048):
        super(TransformerDecorator, self).__init__()
        self.encoder_layers = torch.nn.TransformerEncoderLayer(dim, 4, dim, 0.5)
        self.add_bf = add_bf

    def forward(self, feature):
        if self.training:
            pre_feature = feature
            feature = feature.unsqueeze(1)
            feature = self.encoder_layers(feature)
            feature = feature.squeeze(1)
            return torch.cat([pre_feature, feature], dim=0)
        return feature


"""
add_bf
1 duplicate neg and pos

"""

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, add_bf=0):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        self.add_bf = add_bf
        if self.add_bf:
            self.encoder_global = TransformerDecorator(self.add_bf, 2048, 0)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp), nn.ReLU(), self.encoder_k.fc)

            if 0 < self.add_bf:
                self.encoder_q.fc = nn.Sequential(
                    self.encoder_global,
                    nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp), nn.ReLU(), self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(
                    self.encoder_global,
                    nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp), nn.ReLU(), self.encoder_k.fc)
        assert mlp, "moco v2 should use mlp"


        if self.add_bf:
            for param_q, param_k in zip([k for n, k in self.encoder_q.named_parameters() if not n.__contains__('encoder_layers.')],
                                        [k for n, k in self.encoder_k.named_parameters() if not n.__contains__('encoder_layers.')]):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
        else:
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        if self.add_bf:
            # for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            for param_q, param_k in zip([k for n, k in self.encoder_q.named_parameters() if not n.__contains__('encoder_layers.')],
                                        [k for n, k in self.encoder_k.named_parameters() if not n.__contains__('encoder_layers.')],
                                        ):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        else:
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            old_k = k
            # undo shuffle
            k = self._batch_unshuffle_ddp(old_k[:len(im_q)], idx_unshuffle)
            if self.training and self.add_bf:
                old_k1 = self._batch_unshuffle_ddp(old_k[len(old_k) // 2:], idx_unshuffle)
                k = torch.cat([k, old_k1], dim=0)

        #
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        if self.training and self.add_bf:
            l_pos1 = torch.einsum('nc,nc->n', [q[:len(q)//2], k[:len(q)//2]]).unsqueeze(-1)
            l_pos2 = torch.einsum('nc,nc->n', [q[:len(q)//2], k[len(q)//2:]]).unsqueeze(-1)
            l_pos3 = torch.einsum('nc,nc->n', [q[len(q)//2:], k[:len(q)//2]]).unsqueeze(-1)
            l_pos4 = torch.einsum('nc,nc->n', [q[len(q)//2:], k[len(q)//2:]]).unsqueeze(-1)
            l_pos = torch.cat([l_pos1, l_pos2, l_pos3, l_pos4], dim=0)
        # negative logits: NxK
        if self.training and self.add_bf:
            tmp_q = torch.cat([q[:len(q)//2], q[:len(q)//2], q[len(q)//2:], q[len(q)//2:]], dim=0)
        else:
            tmp_q = q
        l_neg = torch.einsum('nc,ck->nk', [tmp_q, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


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
