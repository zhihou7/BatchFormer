import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .word_embedding import load_word_embeddings
from .common import MLP

device = 'cuda' if torch.cuda.is_available() else 'cpu'



class Symnet(nn.Module):
    def __init__(self, dset, args):
        super(Symnet, self).__init__()
        self.dset = dset
        self.args = args
        self.num_attrs = len(dset.attrs)
        self.num_objs = len(dset.objs)

        self.image_embedder = MLP(dset.feat_dim, args.emb_dim, relu = False)
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.nll_loss = nn.NLLLoss()
        self.softmax = nn.Softmax(dim=1)
        # Attribute embedder and object classifier
        self.attr_embedder = nn.Embedding(len(dset.attrs), args.emb_dim)

        self.obj_classifier = MLP(args.emb_dim, len(dset.objs), num_layers = 2, relu = False, dropout = True, layers = [512]) # original paper uses a second fc classifier for final product
        self.attr_classifier = MLP(args.emb_dim, len(dset.attrs), num_layers = 2, relu = False, dropout = True, layers = [512]) 
        # CoN and DecoN
        self.CoN_fc_attention = MLP(args.emb_dim, args.emb_dim, num_layers = 2, relu = False, dropout = True, layers = [512])
        self.CoN_emb = MLP(args.emb_dim + args.emb_dim, args.emb_dim, num_layers = 2, relu = False, dropout = True, layers = [768])
        self.DecoN_fc_attention = MLP(args.emb_dim, args.emb_dim, num_layers = 2, relu = False, dropout = True, layers = [512])
        self.DecoN_emb = MLP(args.emb_dim + args.emb_dim, args.emb_dim, num_layers = 2, relu = False, dropout = True, layers = [768])

        # if args.glove_init:
        pretrained_weight = load_word_embeddings(args.emb_init, dset.attrs)
        self.attr_embedder.weight.data.copy_(pretrained_weight)

        for param in self.attr_embedder.parameters():
            param.requires_grad = False

    def CoN(self, img_embedding, attr_embedding):
        attention = torch.sigmoid(self.CoN_fc_attention(attr_embedding))
        img_embedding = attention*img_embedding + img_embedding

        hidden = torch.cat([img_embedding, attr_embedding], dim = 1)
        output = self.CoN_emb(hidden)
        return output

    def DeCoN(self, img_embedding, attr_embedding):
        attention = torch.sigmoid(self.DecoN_fc_attention(attr_embedding))
        img_embedding = attention*img_embedding + img_embedding

        hidden = torch.cat([img_embedding, attr_embedding], dim = 1)
        output = self.DecoN_emb(hidden)
        return output


    def distance_metric(self, a, b):
        return torch.norm(a-b, dim = -1)

    def RMD_prob(self, feat_plus, feat_minus, repeat_img_feat):
        """return attribute classification probability with our RMD"""
        # feat_plus, feat_minus:  shape=(bz, #attr, dim_emb)
        # d_plus: distance between feature before&after CoN
        # d_minus: distance between feature before&after DecoN
        d_plus = self.distance_metric(feat_plus, repeat_img_feat).reshape(-1, self.num_attrs)
        d_minus = self.distance_metric(feat_minus, repeat_img_feat).reshape(-1, self.num_attrs)
        # not adding softmax because it is part of cross entropy loss
        return d_minus - d_plus

    def train_forward(self, x):
        pos_image_feat, pos_attr_id, pos_obj_id = x[0], x[1], x[2]
        neg_attr_id = x[4][:,0]

        batch_size = pos_image_feat.size(0)

        loss = []
        pos_attr_emb = self.attr_embedder(pos_attr_id)
        neg_attr_emb = self.attr_embedder(neg_attr_id)

        pos_img = self.image_embedder(pos_image_feat)

        # rA = remove positive attribute A
        # aA = add positive attribute A
        # rB = remove negative attribute B
        # aB = add negative attribute B
        pos_rA = self.DeCoN(pos_img, pos_attr_emb)
        pos_aA = self.CoN(pos_img, pos_attr_emb)
        pos_rB = self.DeCoN(pos_img, neg_attr_emb)
        pos_aB = self.CoN(pos_img, neg_attr_emb)

        # get all attr embedding #attr, embedding
        attr_emb = torch.LongTensor(np.arange(self.num_attrs)).to(device)
        attr_emb = self.attr_embedder(attr_emb)
        tile_attr_emb = attr_emb.repeat(batch_size, 1) # (batch*attr, dim_emb)

        # Now we calculate all the losses
        if self.args.lambda_cls_attr > 0:
            # Original image
            score_pos_A = self.attr_classifier(pos_img)
            loss_cls_pos_a = self.ce_loss(score_pos_A, pos_attr_id)

            # After removing pos_attr
            score_pos_rA_A = self.attr_classifier(pos_rA)
            total = sum(score_pos_rA_A)
            # prob_pos_rA_A = 1 - self.softmax(score_pos_rA_A)
            # loss_cls_pos_rA_a = self.nll_loss(torch.log(prob_pos_rA_A), pos_attr_id)
            loss_cls_pos_rA_a = self.ce_loss(total - score_pos_rA_A, pos_attr_id) #should be maximum for the gt label

            # rmd time
            repeat_img_feat = torch.repeat_interleave(pos_img, self.num_attrs, 0) #(batch*attr, dim_rep)
            feat_plus = self.CoN(repeat_img_feat, tile_attr_emb)
            feat_minus = self.DeCoN(repeat_img_feat, tile_attr_emb)
            score_cls_rmd = self.RMD_prob(feat_plus, feat_minus, repeat_img_feat)
            loss_cls_rmd = self.ce_loss(score_cls_rmd, pos_attr_id)

            loss_cls_attr = self.args.lambda_cls_attr*sum([loss_cls_pos_a, loss_cls_pos_rA_a, loss_cls_rmd])
            loss.append(loss_cls_attr)

        if self.args.lambda_cls_obj > 0:
            # Original image
            score_pos_O = self.obj_classifier(pos_img)
            loss_cls_pos_o = self.ce_loss(score_pos_O, pos_obj_id)

            # After removing pos attr
            score_pos_rA_O = self.obj_classifier(pos_rA)
            loss_cls_pos_rA_o = self.ce_loss(score_pos_rA_O, pos_obj_id)

            # After adding neg attr
            score_pos_aB_O = self.obj_classifier(pos_aB)
            loss_cls_pos_aB_o = self.ce_loss(score_pos_aB_O, pos_obj_id)

            loss_cls_obj = self.args.lambda_cls_obj * sum([loss_cls_pos_o, loss_cls_pos_rA_o, loss_cls_pos_aB_o])

            loss.append(loss_cls_obj)

        if self.args.lambda_sym > 0:

            loss_sys_pos = self.mse_loss(pos_aA, pos_img)
            loss_sys_neg = self.mse_loss(pos_rB, pos_img)
            loss_sym = self.args.lambda_sym * (loss_sys_pos + loss_sys_neg)

            loss.append(loss_sym)

        ##### Axiom losses
        if self.args.lambda_axiom > 0:
            loss_clo = loss_inv = loss_com = 0
            # closure
            pos_aA_rA = self.DeCoN(pos_aA, pos_attr_emb)
            pos_rB_aB = self.CoN(pos_rB, neg_attr_emb)
            loss_clo = self.mse_loss(pos_aA_rA, pos_rA) + self.mse_loss(pos_rB_aB, pos_aB)

            # invertibility
            pos_rA_aA = self.CoN(pos_rA, pos_attr_emb)
            pos_aB_rB = self.DeCoN(pos_aB, neg_attr_emb)
            loss_inv = self.mse_loss(pos_rA_aA, pos_img) + self.mse_loss(pos_aB_rB, pos_img)

            # commutative
            pos_aA_rB = self.DeCoN(pos_aA, neg_attr_emb)
            pos_rB_aA = self.DeCoN(pos_rB, pos_attr_emb)
            loss_com = self.mse_loss(pos_aA_rB, pos_rB_aA)

            loss_axiom = self.args.lambda_axiom * (loss_clo + loss_inv + loss_com)
            loss.append(loss_axiom)
        
        # triplet loss
        if self.args.lambda_trip > 0:
            pos_triplet = F.triplet_margin_loss(pos_img, pos_aA, pos_rA)
            neg_triplet = F.triplet_margin_loss(pos_img, pos_rB, pos_aB)

            loss_triplet = self.args.lambda_trip * (pos_triplet + neg_triplet)
            loss.append(loss_triplet)


        loss = sum(loss)
        return loss, None

    def val_forward(self, x):
        pos_image_feat, pos_attr_id, pos_obj_id = x[0], x[1], x[2]
        batch_size = pos_image_feat.shape[0]
        pos_img = self.image_embedder(pos_image_feat)
        repeat_img_feat = torch.repeat_interleave(pos_img, self.num_attrs, 0) #(batch*attr, dim_rep)
        
        # get all attr embedding #attr, embedding
        attr_emb = torch.LongTensor(np.arange(self.num_attrs)).to(device)
        attr_emb = self.attr_embedder(attr_emb)
        tile_attr_emb = attr_emb.repeat(batch_size, 1) # (batch*attr, dim_emb)

        feat_plus = self.CoN(repeat_img_feat, tile_attr_emb)
        feat_minus = self.DeCoN(repeat_img_feat, tile_attr_emb)
        score_cls_rmd = self.RMD_prob(feat_plus, feat_minus, repeat_img_feat)
        prob_A_rmd = F.softmax(score_cls_rmd, dim = 1)

        score_obj = self.obj_classifier(pos_img)
        prob_O = F.softmax(score_obj, dim = 1)

        scores = {}
        for itr, (attr, obj) in enumerate(self.dset.pairs):
            attr_id, obj_id = self.dset.attr2idx[attr], self.dset.obj2idx[obj]
            score = prob_A_rmd[:,attr_id] * prob_O[:, obj_id]
            
            scores[(attr, obj)] = score
        
        return None, scores

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
        return loss, pred