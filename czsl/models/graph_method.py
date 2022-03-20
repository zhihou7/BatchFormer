import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .common import MLP, cross_domain_triplet_loss
from .gcn import GCN, GCNII

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GraphFull(nn.Module):
    def __init__(self, dset, args):
        super(GraphFull, self).__init__()
        self.args = args
        self.dset = dset

        self.val_forward = self.val_forward_dotpr
        self.train_forward = self.train_forward_normal
        self.add_bt = args.add_bt
        # Image Embedder
        self.num_attrs, self.num_objs, self.num_pairs = len(dset.attrs), len(dset.objs), len(dset.pairs)

        if self.args.train_only:
            train_idx = []
            for current in dset.train_pairs:
                train_idx.append(dset.all_pair2idx[current]+self.num_attrs+self.num_objs)
            self.train_idx = torch.LongTensor(train_idx).to(device)

        if self.args.fc_emb.__contains__(','):
            self.args.fc_emb = self.args.fc_emb.split(',')
        layers = []
        for a in self.args.fc_emb:
            a = int(a)
            layers.append(a)

        if args.nlayers:
            self.image_embedder = MLP(dset.feat_dim, args.emb_dim, num_layers= args.nlayers, dropout = self.args.dropout,
                norm = self.args.norm, layers = layers, relu = True)

        path = args.graph_init
        graph = torch.load(path)
        self.embeddings = graph['embeddings'].to(device)
        adj = graph['adj']

        hidden_layers = self.args.gr_emb

        if args.gcn_type == 'gcn':
            self.gcn = GCN(adj, self.embeddings.shape[1], args.emb_dim, hidden_layers)
        else:
            self.gcn = GCNII(adj, self.embeddings.shape[1], args.emb_dim, args.hidden_dim, args.gcn_nlayers, lamda = 0.5, alpha = 0.1, variant = False)

        if args.static_inp:
            for param in self.parameters():
                    param.requires_grad = False


    def train_forward_normal(self, x):
        img, _, _, pairs = x[0], x[1], x[2], x[3]

        if self.args.nlayers:
            img_feats = self.image_embedder(img)
        else:
            img_feats = (img)

        current_embeddings = self.gcn(self.embeddings)
       
        if self.args.train_only:
            pair_embed = current_embeddings[self.train_idx]
        else:
            pair_embed = current_embeddings[self.num_attrs+self.num_objs:self.num_attrs+self.num_objs+self.num_pairs,:]

        pair_embed = pair_embed.permute(1,0)
        pair_pred = torch.matmul(img_feats, pair_embed)
        if self.add_bt >= 1:
            pairs = torch.cat([pairs, pairs], dim=0)
        loss = F.cross_entropy(pair_pred, pairs)

        return  loss, None

    def val_forward_dotpr(self, x):
        img = x[0]

        if self.args.nlayers:
            img_feats = self.image_embedder(img)
        else:
            img_feats = (img)

        current_embedddings = self.gcn(self.embeddings)

        pair_embeds = current_embedddings[self.num_attrs+self.num_objs:self.num_attrs+self.num_objs+self.num_pairs,:].permute(1,0)

        score = torch.matmul(img_feats, pair_embeds)

        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:,self.dset.all_pair2idx[pair]]

        return None, scores

    def val_forward_distance_fast(self, x):
        img = x[0]

        img_feats = (self.image_embedder(img))
        current_embeddings = self.gcn(self.embeddings)
        pair_embeds = current_embeddings[self.num_attrs+self.num_objs:,:]

        batch_size, pairs, features = img_feats.shape[0], pair_embeds.shape[0], pair_embeds.shape[1]
        img_feats = img_feats[:,None,:].expand(-1, pairs, -1)
        pair_embeds = pair_embeds[None,:,:].expand(batch_size, -1, -1)
        diff = (img_feats - pair_embeds)**2
        score = diff.sum(2) * -1

        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:,self.dset.all_pair2idx[pair]]

        return None, scores

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
        return loss, pred