import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from scipy.stats import hmean

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MLP(nn.Module):
    '''
    Baseclass to create a simple MLP
    Inputs
        inp_dim: Int, Input dimension
        out-dim: Int, Output dimension
        num_layer: Number of hidden layers
        relu: Bool, Use non linear function at output
        bias: Bool, Use bias
    '''
    def __init__(self, inp_dim, out_dim, num_layers = 1, relu = True, bias = True, dropout = False, norm = False, layers = []):
        super(MLP, self).__init__()
        mod = []
        incoming = inp_dim
        for layer in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers.pop(0)
            mod.append(nn.Linear(incoming, outgoing, bias = bias))
            
            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
                # mod.append(nn.BatchNorm1d(outgoing))
            mod.append(nn.ReLU(inplace = True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
            if dropout:
                mod.append(nn.Dropout(p = 0.5))

        mod.append(nn.Linear(incoming, out_dim, bias = bias))

        if relu:
            mod.append(nn.ReLU(inplace = True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
        self.mod = nn.Sequential(*mod)
    
    def forward(self, x):
        return self.mod(x)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

def calculate_margines(domain_embedding, gt, margin_range = 5):
    '''
    domain_embedding: pairs * feats
    gt: batch * feats
    '''
    batch_size, pairs, features = gt.shape[0], domain_embedding.shape[0], domain_embedding.shape[1]
    gt_expanded = gt[:,None,:].expand(-1, pairs, -1)
    domain_embedding_expanded = domain_embedding[None, :, :].expand(batch_size, -1, -1)
    margin = (gt_expanded - domain_embedding_expanded)**2
    margin = margin.sum(2)
    max_margin, _ = torch.max(margin, dim = 0)
    margin /= max_margin
    margin *= margin_range
    return margin

def l2_all_batched(image_embedding, domain_embedding):
    '''
    Image Embedding: Tensor of Batch_size * pairs * Feature_dim
    domain_embedding: Tensor of pairs * Feature_dim
    '''
    pairs = image_embedding.shape[1]
    domain_embedding_extended = image_embedding[:,None,:].expand(-1,pairs,-1)
    l2_loss = (image_embedding - domain_embedding_extended) ** 2
    l2_loss = l2_loss.sum(2)
    l2_loss = l2_loss.sum() / l2_loss.numel()
    return l2_loss

def same_domain_triplet_loss(image_embedding, trip_images, gt, hard_k = None, margin = 2):
    '''
    Image Embedding: Tensor of Batch_size * Feature_dim
    Triplet Images: Tensor of Batch_size * num_pairs * Feature_dim
    GT: Tensor of Batch_size
    '''
    batch_size, pairs, features = trip_images.shape
    batch_iterator = torch.arange(batch_size).to(device)
    image_embedding_expanded = image_embedding[:,None,:].expand(-1, pairs, -1)

    diff = (image_embedding_expanded - trip_images)**2
    diff = diff.sum(2)

    positive_anchor = diff[batch_iterator, gt][:,None]
    positive_anchor = positive_anchor.expand(-1, pairs)

    # Calculating triplet loss
    triplet_loss = positive_anchor - diff + margin
    
    # Setting positive anchor loss to 0
    triplet_loss[batch_iterator, gt] = 0
    
    # Removing easy triplets
    triplet_loss[triplet_loss < 0] = 0

    # If only mining hard triplets
    if hard_k:
        triplet_loss, _ = triplet_loss.topk(hard_k)

    # Counting number of valid pairs
    num_positive_triplets = triplet_loss[triplet_loss > 1e-16].size(0)
    
    # Calculating the final loss
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
    
    return triplet_loss


def cross_domain_triplet_loss(image_embedding, domain_embedding, gt, hard_k = None, margin = 2):
    '''
    Image Embedding: Tensor of Batch_size * Feature_dim
    Domain Embedding: Tensor of Num_pairs * Feature_dim
    gt: Tensor of Batch_size with ground truth labels
    margin: Float of margin
    Returns:
        Triplet loss of all valid triplets
    '''
    batch_size, pairs, features = image_embedding.shape[0], domain_embedding.shape[0], domain_embedding.shape[1]
    batch_iterator = torch.arange(batch_size).to(device)
    # Now dimensions will be Batch_size * Num_pairs * Feature_dim
    image_embedding = image_embedding[:,None,:].expand(-1, pairs, -1)
    domain_embedding = domain_embedding[None,:,:].expand(batch_size, -1, -1)
    
    # Calculating difference
    diff = (image_embedding - domain_embedding)**2
    diff = diff.sum(2)
    
    # Getting the positive pair
    positive_anchor = diff[batch_iterator, gt][:,None]
    positive_anchor = positive_anchor.expand(-1, pairs)
    
    # Calculating triplet loss
    triplet_loss = positive_anchor - diff + margin
    
    # Setting positive anchor loss to 0
    triplet_loss[batch_iterator, gt] = 0
    
    # Removing easy triplets
    triplet_loss[triplet_loss < 0] = 0
    
    # If only mining hard triplets
    if hard_k:
        triplet_loss, _ = triplet_loss.topk(hard_k)

    # Counting number of valid pairs
    num_positive_triplets = triplet_loss[triplet_loss > 1e-16].size(0)
    
    # Calculating the final loss
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
    
    return triplet_loss

def same_domain_triplet_loss_old(image_embedding, positive_anchor, negative_anchor, margin = 2):
    '''
    Image Embedding: Tensor of Batch_size * Feature_dim
    Positive anchor: Tensor of Batch_size * Feature_dim
    negative anchor: Tensor of Batch_size * negs *Feature_dim
    '''
    batch_size, negs, features = negative_anchor.shape
    dist_pos = (image_embedding - positive_anchor)**2
    dist_pos = dist_pos.sum(1)
    dist_pos = dist_pos[:, None].expand(-1, negs)

    image_embedding_expanded = image_embedding[:,None,:].expand(-1, negs, -1)
    dist_neg = (image_embedding_expanded - negative_anchor)**2
    dist_neg = dist_neg.sum(2)

    triplet_loss = dist_pos - dist_neg + margin
    triplet_loss[triplet_loss < 0] = 0
    num_positive_triplets = triplet_loss[triplet_loss > 1e-16].size(0)

    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
    return triplet_loss


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)

class Evaluator:

    def __init__(self, dset, model):

        self.dset = dset

        # Convert text pairs to idx tensors: [('sliced', 'apple'), ('ripe', 'apple'), ...] --> torch.LongTensor([[0,1],[1,1], ...])
        pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.pairs]
        self.train_pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.train_pairs]
        self.pairs = torch.LongTensor(pairs)

        # Mask over pairs that occur in closed world
        # Select set based on phase
        if dset.phase == 'train':
            print('Evaluating with train pairs')
            test_pair_set = set(dset.train_pairs)
            test_pair_gt = set(dset.train_pairs)
        elif dset.phase == 'val':
            print('Evaluating with validation pairs')
            test_pair_set = set(dset.val_pairs + dset.train_pairs)
            test_pair_gt = set(dset.val_pairs)
        else:
            print('Evaluating with test pairs')
            test_pair_set = set(dset.test_pairs + dset.train_pairs)
            test_pair_gt = set(dset.test_pairs)

        self.test_pair_dict = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in test_pair_gt]
        self.test_pair_dict = dict.fromkeys(self.test_pair_dict, 0)

        # dict values are pair val, score, total
        for attr, obj in test_pair_gt:
            pair_val = dset.pair2idx[(attr,obj)]
            key = (dset.attr2idx[attr], dset.obj2idx[obj])
            self.test_pair_dict[key] = [pair_val, 0, 0]

        if dset.open_world:
            masks = [1 for _ in dset.pairs]
        else:
            masks = [1 if pair in test_pair_set else 0 for pair in dset.pairs]

        self.closed_mask = torch.BoolTensor(masks)
        # Mask of seen concepts
        seen_pair_set = set(dset.train_pairs)
        mask = [1 if pair in seen_pair_set  else 0 for pair in dset.pairs]
        self.seen_mask = torch.BoolTensor(mask)

        # Object specific mask over which pairs occur in the object oracle setting
        oracle_obj_mask = []
        for _obj in dset.objs:
            mask = [1  if _obj == obj else 0 for attr, obj in dset.pairs]
            oracle_obj_mask.append(torch.BoolTensor(mask))
        self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)

        # Decide if the model under evaluation is a manifold model or not
        self.score_model = self.score_manifold_model

    # Generate mask for each settings, mask scores, and get prediction labels
    def generate_predictions(self, scores, obj_truth, bias = 0.0, topk = 5): # (Batch, #pairs)
        '''
        Inputs
            scores: Output scores
            obj_truth: Ground truth object
        Returns
            results: dict of results in 3 settings
        '''
        def get_pred_from_scores(_scores, topk):
            '''
            Given list of scores, returns top 10 attr and obj predictions
            Check later
            '''
            _, pair_pred = _scores.topk(topk, dim = 1) #sort returns indices of k largest values
            pair_pred = pair_pred.contiguous().view(-1)
            attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(-1, topk), \
                self.pairs[pair_pred][:, 1].view(-1, topk)
            return (attr_pred, obj_pred)

        results = {}
        orig_scores = scores.clone()
        mask = self.seen_mask.repeat(scores.shape[0],1) # Repeat mask along pairs dimension
        scores[~mask] += bias # Add bias to test pairs

        # Unbiased setting
        
        # Open world setting --no mask, all pairs of the dataset
        results.update({'open': get_pred_from_scores(scores, topk)})
        results.update({'unbiased_open': get_pred_from_scores(orig_scores, topk)})
        # Closed world setting - set the score for all Non test pairs to -1e10, 
        # this excludes the pairs from set not in evaluation
        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10 
        closed_orig_scores = orig_scores.clone()
        closed_orig_scores[~mask] = -1e10
        results.update({'closed': get_pred_from_scores(closed_scores, topk)})
        results.update({'unbiased_closed': get_pred_from_scores(closed_orig_scores, topk)})

        # Object_oracle setting - set the score to -1e10 for all pairs where the true object does Not participate, can also use the closed score
        mask = self.oracle_obj_mask[obj_truth]
        oracle_obj_scores = scores.clone()
        oracle_obj_scores[~mask] = -1e10
        oracle_obj_scores_unbiased = orig_scores.clone()
        oracle_obj_scores_unbiased[~mask] = -1e10
        results.update({'object_oracle': get_pred_from_scores(oracle_obj_scores, 1)})
        results.update({'object_oracle_unbiased': get_pred_from_scores(oracle_obj_scores_unbiased, 1)})

        return results

    def score_clf_model(self, scores, obj_truth, topk = 5):
        '''
        Wrapper function to call generate_predictions for CLF models
        '''
        attr_pred, obj_pred = scores

        # Go to CPU
        attr_pred, obj_pred, obj_truth = attr_pred.to('cpu'), obj_pred.to('cpu'), obj_truth.to('cpu')

        # Gather scores (P(a), P(o)) for all relevant (a,o) pairs
        # Multiply P(a) * P(o) to get P(pair)
        attr_subset = attr_pred.index_select(1, self.pairs[:,0]) # Return only attributes that are in our pairs
        obj_subset = obj_pred.index_select(1, self.pairs[:, 1])
        scores = (attr_subset * obj_subset) # (Batch, #pairs)

        results = self.generate_predictions(scores, obj_truth)
        results['biased_scores'] = scores

        return results

    def score_manifold_model(self, scores, obj_truth, bias = 0.0, topk = 5):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''
        # Go to CPU
        scores = {k: v.to('cpu') for k, v in scores.items()}
        obj_truth = obj_truth.to(device)

        # Gather scores for all relevant (a,o) pairs
        scores = torch.stack(
            [scores[(attr,obj)] for attr, obj in self.dset.pairs], 1
        ) # (Batch, #pairs)
        orig_scores = scores.clone()
        results = self.generate_predictions(scores, obj_truth, bias, topk)
        results['scores'] = orig_scores
        return results

    def score_fast_model(self, scores, obj_truth, bias = 0.0, topk = 5):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''
        
        results = {}
        mask = self.seen_mask.repeat(scores.shape[0],1) # Repeat mask along pairs dimension
        scores[~mask] += bias # Add bias to test pairs

        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10 

        _, pair_pred = closed_scores.topk(topk, dim = 1) #sort returns indices of k largest values
        pair_pred = pair_pred.contiguous().view(-1)
        attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(-1, topk), \
            self.pairs[pair_pred][:, 1].view(-1, topk)

        results.update({'closed': (attr_pred, obj_pred)})
        return results

    def evaluate_predictions(self, predictions, attr_truth, obj_truth, pair_truth, allpred, topk = 1):
        # Go to CPU
        attr_truth, obj_truth, pair_truth = attr_truth.to('cpu'), obj_truth.to('cpu'), pair_truth.to('cpu')

        pairs = list(
            zip(list(attr_truth.numpy()), list(obj_truth.numpy())))
        

        seen_ind, unseen_ind = [], []
        for i in range(len(attr_truth)):
            if pairs[i] in self.train_pairs:
                seen_ind.append(i)
            else:
                unseen_ind.append(i)

        
        seen_ind, unseen_ind = torch.LongTensor(seen_ind), torch.LongTensor(unseen_ind)
        def _process(_scores):
            # Top k pair accuracy
            # Attribute, object and pair
            attr_match = (attr_truth.unsqueeze(1).repeat(1, topk) == _scores[0][:, :topk])
            obj_match = (obj_truth.unsqueeze(1).repeat(1, topk) == _scores[1][:, :topk])

            # Match of object pair
            match = (attr_match * obj_match).any(1).float()
            attr_match = attr_match.any(1).float()
            obj_match = obj_match.any(1).float()
            # Match of seen and unseen pairs
            seen_match = match[seen_ind]
            unseen_match = match[unseen_ind]
            ### Calculating class average accuracy
            
            # local_score_dict = copy.deepcopy(self.test_pair_dict)
            # for pair_gt, pair_pred in zip(pairs, match):
            #     # print(pair_gt)
            #     local_score_dict[pair_gt][2] += 1.0 #increase counter
            #     if int(pair_pred) == 1:
            #         local_score_dict[pair_gt][1] += 1.0

            # # Now we have hits and totals for classes in evaluation set
            # seen_score, unseen_score = [], []
            # for key, (idx, hits, total) in local_score_dict.items():
            #     score = hits/total
            #     if bool(self.seen_mask[idx]) == True:
            #         seen_score.append(score)
            #     else:
            #         unseen_score.append(score)
            
            seen_score, unseen_score = torch.ones(512,5), torch.ones(512,5)

            return attr_match, obj_match, match, seen_match, unseen_match, \
            torch.Tensor(seen_score+unseen_score), torch.Tensor(seen_score), torch.Tensor(unseen_score)

        def _add_to_dict(_scores, type_name, stats):
            base = ['_attr_match', '_obj_match', '_match', '_seen_match', '_unseen_match', '_ca', '_seen_ca', '_unseen_ca']
            for val, name in zip(_scores, base):
                stats[type_name + name] = val

        ##################### Match in places where corrent object
        obj_oracle_match = (attr_truth == predictions['object_oracle'][0][:, 0]).float()  #object is already conditioned
        obj_oracle_match_unbiased = (attr_truth == predictions['object_oracle_unbiased'][0][:, 0]).float()

        stats = dict(obj_oracle_match = obj_oracle_match, obj_oracle_match_unbiased = obj_oracle_match_unbiased)

        #################### Closed world
        closed_scores = _process(predictions['closed'])
        unbiased_closed = _process(predictions['unbiased_closed'])
        _add_to_dict(closed_scores, 'closed', stats)
        _add_to_dict(unbiased_closed, 'closed_ub', stats)

        #################### Calculating AUC
        scores = predictions['scores']
        # getting score for each ground truth class
        correct_scores = scores[torch.arange(scores.shape[0]), pair_truth][unseen_ind]

        # Getting top predicted score for these unseen classes
        max_seen_scores = predictions['scores'][unseen_ind][:, self.seen_mask].topk(topk, dim=1)[0][:, topk - 1]

        # Getting difference between these scores
        unseen_score_diff = max_seen_scores - correct_scores

        # Getting matched classes at max bias for diff
        unseen_matches = stats['closed_unseen_match'].bool()
        correct_unseen_score_diff = unseen_score_diff[unseen_matches] - 1e-4

        # sorting these diffs
        correct_unseen_score_diff = torch.sort(correct_unseen_score_diff)[0]
        magic_binsize = 20
        # getting step size for these bias values
        bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
        # Getting list
        biaslist = correct_unseen_score_diff[::bias_skip]

        seen_match_max = float(stats['closed_seen_match'].mean())
        unseen_match_max = float(stats['closed_unseen_match'].mean())
        seen_accuracy, unseen_accuracy = [], []

        # Go to CPU
        base_scores = {k: v.to('cpu') for k, v in allpred.items()}
        obj_truth = obj_truth.to('cpu')

        # Gather scores for all relevant (a,o) pairs
        base_scores = torch.stack(
            [allpred[(attr,obj)] for attr, obj in self.dset.pairs], 1
        ) # (Batch, #pairs)

        for bias in biaslist:
            scores = base_scores.clone()
            results = self.score_fast_model(scores, obj_truth, bias = bias, topk = topk)
            results = results['closed'] # we only need biased
            results = _process(results)
            seen_match = float(results[3].mean())
            unseen_match = float(results[4].mean())
            seen_accuracy.append(seen_match)
            unseen_accuracy.append(unseen_match)

        seen_accuracy.append(seen_match_max)
        unseen_accuracy.append(unseen_match_max)
        seen_accuracy, unseen_accuracy = np.array(seen_accuracy), np.array(unseen_accuracy)
        area = np.trapz(seen_accuracy, unseen_accuracy)

        for key in stats:
            stats[key] = float(stats[key].mean())

        harmonic_mean = hmean([seen_accuracy, unseen_accuracy], axis = 0)
        max_hm = np.max(harmonic_mean)
        idx = np.argmax(harmonic_mean)
        if idx == len(biaslist):
            bias_term = 1e3
        else:
            bias_term = biaslist[idx]
        stats['biasterm'] = float(bias_term)
        stats['best_unseen'] = np.max(unseen_accuracy)
        stats['best_seen'] = np.max(seen_accuracy)
        stats['AUC'] = area
        stats['hm_unseen'] = unseen_accuracy[idx]
        stats['hm_seen'] = seen_accuracy[idx]
        stats['best_hm'] = max_hm
        return stats