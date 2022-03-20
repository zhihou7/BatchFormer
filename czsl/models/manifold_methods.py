import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .word_embedding import load_word_embeddings
from .common import MLP, Reshape
from flags import DATA_FOLDER

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ManifoldModel(nn.Module):

    def __init__(self, dset, args):
        super(ManifoldModel, self).__init__()
        self.args = args
        self.dset = dset

        def get_all_ids(relevant_pairs):
            # Precompute validation pairs
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            attrs = torch.LongTensor(attrs).to(device)
            objs = torch.LongTensor(objs).to(device)
            pairs = torch.LongTensor(pairs).to(device)
            return attrs, objs, pairs

        #Validation
        self.val_attrs, self.val_objs, self.val_pairs = get_all_ids(self.dset.pairs)
        # for indivual projections
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(device), \
                                            torch.arange(len(self.dset.objs)).long().to(device)
        self.factor = 2
        # Precompute training compositions
        if args.train_only:
            self.train_attrs, self.train_objs, self.train_pairs = get_all_ids(self.dset.train_pairs)
        else:
            self.train_attrs, self.train_objs = self.val_subs, self.val_attrs, self.val_objs

        if args.lambda_aux > 0 or args.lambda_cls_attr > 0 or args.lambda_cls_obj > 0:
            print('Initializing classifiers')
            self.obj_clf = nn.Linear(args.emb_dim, len(dset.objs))
            self.attr_clf = nn.Linear(args.emb_dim, len(dset.attrs))

    def train_forward_bce(self, x):
        img, attrs, objs = x[0], x[1], x[2]
        neg_attrs, neg_objs = x[4][:,0],x[5][:,0] #todo: do a less hacky version
        
        img_feat = self.image_embedder(img)
        # Sample 25% positive and 75% negative pairs
        labels = np.random.binomial(1, 0.25, attrs.shape[0])
        labels = torch.from_numpy(labels).bool().to(device)
        sampled_attrs, sampled_objs = neg_attrs.clone(), neg_objs.clone()
        sampled_attrs[labels] = attrs[labels]
        sampled_objs[labels] = objs[labels]
        labels = labels.float()

        composed_clf = self.compose(attrs, objs)
        p = torch.sigmoid((img_feat*composed_clf).sum(1))
        loss = F.binary_cross_entropy(p, labels)

        return loss, None

    def train_forward_triplet(self, x):
        img, attrs, objs = x[0], x[1], x[2]
        neg_attrs, neg_objs = x[4][:,0], x[5][:,0] #todo:do a less hacky version

        img_feats = self.image_embedder(img)
        positive = self.compose(attrs, objs)
        negative = self.compose(neg_attrs, neg_objs)
        loss = F.triplet_margin_loss(img_feats, positive, negative,  margin = self.args.margin)

        # Auxiliary object/ attribute prediction loss both need to be correct
        if self.args.lambda_aux > 0:
            obj_pred = self.obj_clf(positive)
            attr_pred = self.attr_clf(positive)
            loss_aux = F.cross_entropy(attr_pred, attrs) + F.cross_entropy(obj_pred, objs)
            loss += self.args.lambda_aux * loss_aux

        return  loss, None


    def val_forward_distance(self, x):
        img = x[0]
        batch_size = img.shape[0]

        img_feats = self.image_embedder(img)
        scores = {}
        pair_embeds = self.compose(self.val_attrs, self.val_objs)
        
        for itr, pair in enumerate(self.dset.pairs):
            pair_embed = pair_embeds[itr, None].expand(batch_size, pair_embeds.size(1))
            score = self.compare_metric(img_feats, pair_embed)
            scores[pair] = score

        return None, scores

    def val_forward_distance_fast(self, x):
        img = x[0]
        batch_size = img.shape[0]

        img_feats = self.image_embedder(img)
        pair_embeds = self.compose(self.val_attrs, self.val_objs) # Evaluate all pairs

        batch_size, pairs, features = img_feats.shape[0], pair_embeds.shape[0], pair_embeds.shape[1]
        img_feats = img_feats[:,None,:].expand(-1, pairs, -1)
        pair_embeds = pair_embeds[None,:,:].expand(batch_size, -1, -1)
        diff = (img_feats - pair_embeds)**2
        score = diff.sum(2) * -1

        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:,self.dset.all_pair2idx[pair]]

        return None, scores
    
    def val_forward_direct(self, x):
        img = x[0]
        batch_size = img.shape[0]

        img_feats = self.image_embedder(img)
        pair_embeds = self.compose(self.val_attrs, self.val_objs).permute(1,0) # Evaluate all pairs
        score = torch.matmul(img_feats, pair_embeds)

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

#########################################
class RedWine(ManifoldModel):

    def __init__(self, dset, args):
        super(RedWine, self).__init__(dset, args)
        self.image_embedder = lambda img: img
        self.compare_metric = lambda img_feats, pair_embed: torch.sigmoid((img_feats*pair_embed).sum(1))
        self.train_forward = self.train_forward_bce
        self.val_forward = self.val_forward_distance

        in_dim = dset.feat_dim if not args.glove_init else 300
        self.T = nn.Sequential(
            nn.Linear(2*in_dim, 3*in_dim),
            nn.LeakyReLU(0.1, True),
            nn.Linear(3*in_dim, 3*in_dim//2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(3*in_dim//2, dset.feat_dim))

        self.attr_embedder = nn.Embedding(len(dset.attrs), in_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), in_dim)

        # initialize the weights of the embedders with the svm weights
        if args.glove_init:
            pretrained_weight = load_word_embeddings(DATA_FOLDER+'/glove/glove.6B.300d.txt', dset.attrs)
            self.attr_embedder.weight.data.copy_(pretrained_weight)
            pretrained_weight = load_word_embeddings(DATA_FOLDER+'/glove/glove.6B.300d.txt', dset.objs)
            self.obj_embedder.weight.data.copy_(pretrained_weight)

        elif args.clf_init:
            for idx, attr in enumerate(dset.attrs):
                at_id = self.dset.attr2idx[attr]
                weight = torch.load('%s/svm/attr_%d'%(args.data_dir, at_id)).coef_.squeeze()
                self.attr_embedder.weight[idx].data.copy_(torch.from_numpy(weight))
            for idx, obj in enumerate(dset.objs):
                obj_id = self.dset.obj2idx[obj]
                weight = torch.load('%s/svm/obj_%d'%(args.data_dir, obj_id)).coef_.squeeze()
                self.obj_embedder.weight[idx].data.copy_(torch.from_numpy(weight))
        else:
            print ('init must be either glove or clf')
            return

        if args.static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

    def compose(self, attrs, objs):
        attr_wt = self.attr_embedder(attrs)
        obj_wt = self.obj_embedder(objs)
        inp_wts = torch.cat([attr_wt, obj_wt], 1) # 2D
        composed_clf = self.T(inp_wts)
        return composed_clf

class LabelEmbedPlus(ManifoldModel):
    def __init__(self, dset, args):
        super(LabelEmbedPlus, self).__init__(dset, args)
        if 'conv' in args.image_extractor:
            self.image_embedder = torch.nn.Sequential(torch.nn.Conv2d(dset.feat_dim,args.emb_dim,7),
                      torch.nn.ReLU(True),
                      Reshape(-1,args.emb_dim)
                      )
        else:
            self.image_embedder = MLP(dset.feat_dim, args.emb_dim)

        self.compare_metric = lambda img_feats, pair_embed: -F.pairwise_distance(img_feats, pair_embed)
        self.train_forward = self.train_forward_triplet
        self.val_forward = self.val_forward_distance_fast

        input_dim = dset.feat_dim if args.clf_init else args.emb_dim
        self.attr_embedder = nn.Embedding(len(dset.attrs), input_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), input_dim)
        self.T = MLP(2*input_dim, args.emb_dim, num_layers= args.nlayers)

        # init with word embeddings
        if args.emb_init:
            pretrained_weight = load_word_embeddings(args.emb_init, dset.attrs)
            self.attr_embedder.weight.data.copy_(pretrained_weight)
            pretrained_weight = load_word_embeddings(args.emb_init, dset.objs)
            self.obj_embedder.weight.data.copy_(pretrained_weight)

        # init with classifier weights
        elif args.clf_init:
            for idx, attr in enumerate(dset.attrs):
                at_id = dset.attrs.index(attr)
                weight = torch.load('%s/svm/attr_%d'%(args.data_dir, at_id)).coef_.squeeze()
                self.attr_embedder.weight[idx].data.copy_(torch.from_numpy(weight))
            for idx, obj in enumerate(dset.objs):
                obj_id = dset.objs.index(obj)
                weight = torch.load('%s/svm/obj_%d'%(args.data_dir, obj_id)).coef_.squeeze()
                self.obj_emb.weight[idx].data.copy_(torch.from_numpy(weight))

        # static inputs
        if args.static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

    def compose(self, attrs, objs):
        inputs = [self.attr_embedder(attrs), self.obj_embedder(objs)]
        inputs = torch.cat(inputs, 1)
        output = self.T(inputs)
        return output

class AttributeOperator(ManifoldModel):
    def __init__(self, dset, args):
        super(AttributeOperator, self).__init__(dset, args)
        self.image_embedder = MLP(dset.feat_dim, args.emb_dim)
        self.compare_metric = lambda img_feats, pair_embed: -F.pairwise_distance(img_feats, pair_embed)
        self.val_forward = self.val_forward_distance_fast

        self.attr_ops = nn.ParameterList([nn.Parameter(torch.eye(args.emb_dim)) for _ in range(len(self.dset.attrs))])
        self.obj_embedder = nn.Embedding(len(dset.objs), args.emb_dim)

        if args.emb_init:
            pretrained_weight = load_word_embeddings('glove', dset.objs)
            self.obj_embedder.weight.data.copy_(pretrained_weight)
            
        self.inverse_cache = {}

        if args.lambda_ant>0 and args.dataset=='mitstates':
            antonym_list = open(DATA_FOLDER+'/data/antonyms.txt').read().strip().split('\n')
            antonym_list = [l.split() for l in antonym_list]
            antonym_list = [[self.dset.attrs.index(a1), self.dset.attrs.index(a2)] for a1, a2 in antonym_list]
            antonyms = {}
            antonyms.update({a1:a2 for a1, a2 in antonym_list})
            antonyms.update({a2:a1 for a1, a2 in antonym_list})
            self.antonyms, self.antonym_list = antonyms, antonym_list

        if args.static_inp:
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

    def apply_ops(self, ops, rep):
        out = torch.bmm(ops, rep.unsqueeze(2)).squeeze(2)
        out = F.relu(out)
        return out

    def compose(self, attrs, objs):
        obj_rep = self.obj_embedder(objs)
        attr_ops = torch.stack([self.attr_ops[attr.item()] for attr in attrs])
        embedded_reps = self.apply_ops(attr_ops, obj_rep)
        return embedded_reps

    def apply_inverse(self, img_rep, attrs):
        inverse_ops = []
        for i in range(img_rep.size(0)):
            attr = attrs[i]
            if attr not in self.inverse_cache:
                self.inverse_cache[attr] = self.attr_ops[attr].inverse()
            inverse_ops.append(self.inverse_cache[attr])
        inverse_ops = torch.stack(inverse_ops) # (B,512,512)
        obj_rep = self.apply_ops(inverse_ops, img_rep)
        return obj_rep

    def train_forward(self, x):
        img, attrs, objs = x[0], x[1], x[2]
        neg_attrs, neg_objs, inv_attrs, comm_attrs = x[4][:,0], x[5][:,0], x[6], x[7]
        batch_size = img.size(0)

        loss = []

        anchor = self.image_embedder(img)

        obj_emb = self.obj_embedder(objs)
        pos_ops = torch.stack([self.attr_ops[attr.item()] for attr in attrs])
        positive = self.apply_ops(pos_ops, obj_emb)

        neg_obj_emb = self.obj_embedder(neg_objs)
        neg_ops = torch.stack([self.attr_ops[attr.item()] for attr in neg_attrs])
        negative = self.apply_ops(neg_ops, neg_obj_emb)

        loss_triplet = F.triplet_margin_loss(anchor, positive, negative, margin=self.args.margin)
        loss.append(loss_triplet)


        # Auxiliary object/attribute loss
        if self.args.lambda_aux>0:
            obj_pred = self.obj_clf(positive)
            attr_pred = self.attr_clf(positive)
            loss_aux = F.cross_entropy(attr_pred, attrs) + F.cross_entropy(obj_pred, objs)
            loss.append(self.args.lambda_aux*loss_aux)

        # Inverse Consistency
        if self.args.lambda_inv>0:
            obj_rep = self.apply_inverse(anchor, attrs)
            new_ops = torch.stack([self.attr_ops[attr.item()] for attr in inv_attrs])
            new_rep = self.apply_ops(new_ops, obj_rep)
            new_positive = self.apply_ops(new_ops, obj_emb)
            loss_inv = F.triplet_margin_loss(new_rep, new_positive, positive, margin=self.args.margin)
            loss.append(self.args.lambda_inv*loss_inv)

        # Commutative Operators
        if self.args.lambda_comm>0:
            B = torch.stack([self.attr_ops[attr.item()] for attr in comm_attrs])
            BA = self.apply_ops(B, positive)
            AB = self.apply_ops(pos_ops, self.apply_ops(B, obj_emb))
            loss_comm = ((AB-BA)**2).sum(1).mean()
            loss.append(self.args.lambda_comm*loss_comm)

        # Antonym Consistency
        if self.args.lambda_ant>0:

            select_idx = [i for i in range(batch_size) if attrs[i].item() in self.antonyms]
            if len(select_idx)>0:
                select_idx = torch.LongTensor(select_idx).cuda()
                attr_subset = attrs[select_idx]
                antonym_ops = torch.stack([self.attr_ops[self.antonyms[attr.item()]] for attr in attr_subset])

                Ao = anchor[select_idx]
                if self.args.lambda_inv>0:
                    o = obj_rep[select_idx]
                else:
                    o = self.apply_inverse(Ao, attr_subset)
                BAo = self.apply_ops(antonym_ops, Ao)

                loss_cycle = ((BAo-o)**2).sum(1).mean()
                loss.append(self.args.lambda_ant*loss_cycle)


        loss = sum(loss)
        return loss, None

    def forward(self, x):
        loss, pred = super(AttributeOperator, self).forward(x)
        self.inverse_cache = {}
        return loss, pred