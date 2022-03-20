import numpy as np
import tqdm
from data import dataset as dset
import os
from utils import utils
import torch
from torch.autograd import Variable
import h5py
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import torch.nn.functional as F
from joblib import Parallel, delayed
import glob
import scipy.io
from sklearn.calibration import CalibratedClassifierCV


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--dataset', default='mitstates', help='mitstates|zappos')
parser.add_argument('--data_dir', default='data/mit-states/')
parser.add_argument('--generate', action ='store_true', default=False)
parser.add_argument('--evalsvm', action ='store_true', default=False)
parser.add_argument('--evaltf', action ='store_true', default=False)
parser.add_argument('--completed', default='tensor-completion/completed/complete.mat')
args = parser.parse_args()

#----------------------------------------------------------------------------------------#

search_params = {'C': np.logspace(-5,5,11)}
def train_svm(Y_train, sample_weight=None):
    
    try:
        clf = GridSearchCV(LinearSVC(class_weight='balanced', fit_intercept=False), search_params, scoring='f1', cv=4)
        clf.fit(X_train, Y_train, sample_weight=sample_weight)
    except:
        clf = LinearSVC(C=0.1, class_weight='balanced', fit_intercept=False)
        if Y_train.sum()==len(Y_train) or Y_train.sum()==0:
            return None
        clf.fit(X_train, Y_train, sample_weight=sample_weight)

    return clf

def generate_svms():
    # train an SVM for every attribute, object and pair primitive

    Y = [(train_attrs==attr).astype(np.int) for attr in range(len(dataset.attrs))]
    attr_clfs = Parallel(n_jobs=32, verbose=16)(delayed(train_svm)(Y[attr]) for attr in range(len(dataset.attrs)))
    for attr, clf in enumerate(attr_clfs):
        print (attr, dataset.attrs[attr])
        print ('params:', clf.best_params_)
        print ('-'*30)
        torch.save(clf.best_estimator_, '%s/svm/attr_%d'%(args.data_dir, attr))

    Y = [(train_objs==obj).astype(np.int) for obj in range(len(dataset.objs))]
    obj_clfs = Parallel(n_jobs=32, verbose=16)(delayed(train_svm)(Y[obj]) for obj in range(len(dataset.objs)))
    for obj, clf in enumerate(obj_clfs):
        print (obj, dataset.objs[obj])
        print ('params:', clf.best_params_)
        print ('-'*30)
        torch.save(clf.best_estimator_, '%s/svm/obj_%d'%(args.data_dir, obj))


    Y, Y_attr, sample_weight = [], [], []
    for idx, (attr, obj) in enumerate(dataset.train_pairs):
        Y_train = ((train_attrs==attr)*(train_objs==obj)).astype(np.int)

        # reweight instances to get a little more training data
        Y_train_attr = (train_attrs==attr).astype(np.int)
        instance_weights = 0.1*Y_train_attr
        instance_weights[Y_train.nonzero()[0]] = 1.0

        Y.append(Y_train)
        Y_attr.append(Y_train_attr)
        sample_weight.append(instance_weights)

    pair_clfs = Parallel(n_jobs=32, verbose=16)(delayed(train_svm)(Y[pair]) for pair in range(len(dataset.train_pairs)))
    for idx, (attr, obj) in enumerate(dataset.train_pairs):
        clf = pair_clfs[idx]
        print (dataset.attrs[attr], dataset.objs[obj])

        try:
            print ('params:', clf.best_params_)
            torch.save(clf.best_estimator_, '%s/svm/pair_%d_%d'%(args.data_dir, attr, obj))
        except:
            print ('FAILED! #positive:', Y[idx].sum(), len(Y[idx]))

    return

def make_svm_tensor():
    subs, vals, size = [], [], (len(dataset.attrs), len(dataset.objs), X.shape[1])
    fullsubs, fullvals = [], []
    composite_clfs = glob.glob('%s/svm/pair*'%args.data_dir)
    print ('%d composite classifiers found'%(len(composite_clfs)))
    for clf in tqdm.tqdm(composite_clfs):
        _, attr, obj = os.path.basename(clf).split('_')
        attr, obj = int(attr), int(obj)

        clf = torch.load(clf)
        weight = clf.coef_.squeeze()
        for i in range(len(weight)):
            subs.append((attr, obj, i))
            vals.append(weight[i])

    for attr, obj in dataset.pairs:
        for i in range(X.shape[1]):
            fullsubs.append((attr, obj, i))

    subs, vals = np.array(subs), np.array(vals).reshape(-1,1)
    fullsubs, fullvals = np.array(fullsubs), np.ones(len(fullsubs)).reshape(-1,1)
    savedat = {'subs':subs, 'vals':vals, 'size':size, 'fullsubs':fullsubs, 'fullvals':fullvals}
    scipy.io.savemat('tensor-completion/incomplete/%s.mat'%args.dataset, savedat)
    print (subs.shape, vals.shape, size)

def evaluate_svms():

    attr_clfs = [torch.load('%s/svm/attr_%d'%(args.data_dir, attr)) for attr in range(len(dataset.attrs))]
    obj_clfs = [torch.load('%s/svm/obj_%d'%(args.data_dir, obj)) for obj in range(len(dataset.objs))]

    # Calibrate all classifiers first
    Y = [(train_attrs==attr).astype(np.int) for attr in range(len(dataset.attrs))]
    for attr in tqdm.tqdm(range(len(dataset.attrs))):
        clf = attr_clfs[attr]
        calibrated = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
        calibrated.fit(X_train, Y[attr])
        attr_clfs[attr] = calibrated

    Y = [(train_objs==obj).astype(np.int) for obj in range(len(dataset.objs))]
    for obj in tqdm.tqdm(range(len(dataset.objs))):
        clf = obj_clfs[obj]
        calibrated = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
        calibrated.fit(X_train, Y[obj])
        obj_clfs[obj] = calibrated

    # Generate all the scores
    attr_scores, obj_scores = [], []
    for attr in tqdm.tqdm(range(len(dataset.attrs))):
        clf = attr_clfs[attr]
        score = clf.predict_proba(X_test)[:,1]
        attr_scores.append(score)
    attr_scores = np.vstack(attr_scores)

    for obj in tqdm.tqdm(range(len(dataset.objs))):
        clf = obj_clfs[obj]
        score = clf.predict_proba(X_test)[:,1]
        obj_scores.append(score)
    obj_scores = np.vstack(obj_scores)

    attr_pred = torch.from_numpy(attr_scores).transpose(0,1)
    obj_pred = torch.from_numpy(obj_scores).transpose(0,1)

    x = [None, Variable(torch.from_numpy(test_attrs)).long(), Variable(torch.from_numpy(test_objs)).long(), Variable(torch.from_numpy(test_pairs)).long()]
    attr_pred, obj_pred, _ = utils.generate_prediction_tensors([attr_pred, obj_pred], dataset, x[2].data, source='classification')
    attr_match, obj_match, zsl_match, gzsl_match, fixobj_match = utils.performance_stats(attr_pred, obj_pred, x)
    print (attr_match.mean(), obj_match.mean(), zsl_match.mean(), gzsl_match.mean(), fixobj_match.mean())

def evaluate_tensorcompletion():

    def parse_tensor(fl):
        tensor = scipy.io.loadmat(fl)
        nz_idx = zip(*(tensor['subs']))
        composite_clfs = np.zeros((len(dataset.attrs), len(dataset.objs), X.shape[1]))
        composite_clfs[nz_idx[0], nz_idx[1], nz_idx[2]] = tensor['vals'].squeeze()
        return composite_clfs, nz_idx, tensor['vals'].squeeze()

    # see recon error
    tr_file = 'tensor-completion/incomplete/%s.mat'%args.dataset
    ts_file = args.completed

    tr_clfs, tr_nz_idx, tr_vals = parse_tensor(tr_file)
    ts_clfs, ts_nz_idx, ts_vals = parse_tensor(ts_file)

    print (tr_vals.min(), tr_vals.max(), tr_vals.mean())
    print (ts_vals.min(), ts_vals.max(), ts_vals.mean())

    print ('Completed Tensor: %s'%args.completed)

    # see train recon error
    err = 1.0*((tr_clfs[tr_nz_idx[0], tr_nz_idx[1], tr_nz_idx[2]]-ts_clfs[tr_nz_idx[0], tr_nz_idx[1], tr_nz_idx[2]])**2).sum()/(len(tr_vals))
    print ('recon error:', err)

    # Create and scale classifiers for each pair
    clfs = {}
    test_pair_set = set(map(tuple, dataset.test_pairs.numpy().tolist()))
    for idx, (attr, obj) in tqdm.tqdm(enumerate(dataset.pairs), total=len(dataset.pairs)):
        clf = LinearSVC(fit_intercept=False)
        clf.fit(np.eye(2), [0,1])

        if (attr, obj) in test_pair_set:
            X_ = X_test
            Y_ = (test_attrs==attr).astype(np.int)*(test_objs==obj).astype(np.int)
            clf.coef_ = ts_clfs[attr, obj][None,:]
        else:
            X_ = X_train
            Y_ = (train_attrs==attr).astype(np.int)*(train_objs==obj).astype(np.int)
            clf.coef_ = tr_clfs[attr, obj][None,:]
        
        calibrated = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
        calibrated.fit(X_, Y_)
        clfs[(attr, obj)] = calibrated

    scores = {}
    for attr, obj in tqdm.tqdm(dataset.pairs):
        score = clfs[(attr, obj)].predict_proba(X_test)[:,1]
        scores[(attr, obj)] = torch.from_numpy(score).float().unsqueeze(1)

    x = [None, Variable(torch.from_numpy(test_attrs)).long(), Variable(torch.from_numpy(test_objs)).long(), Variable(torch.from_numpy(test_pairs)).long()]
    attr_pred, obj_pred, _ = utils.generate_prediction_tensors(scores, dataset, x[2].data, source='manifold')
    attr_match, obj_match, zsl_match, gzsl_match, fixobj_match = utils.performance_stats(attr_pred, obj_pred, x)
    print (attr_match.mean(), obj_match.mean(), zsl_match.mean(), gzsl_match.mean(), fixobj_match.mean())



#----------------------------------------------------------------------------------------#
if args.dataset == 'mitstates':
    DSet = dset.MITStatesActivations
elif args.dataset == 'zappos':
    DSet = dset.UTZapposActivations

dataset = DSet(root=args.data_dir, phase='train')
train_idx, train_attrs, train_objs, train_pairs = map(np.array, zip(*dataset.train_data))
test_idx, test_attrs, test_objs, test_pairs = map(np.array, zip(*dataset.test_data))

X = dataset.activations.numpy()
X_train, X_test = X[train_idx,:], X[test_idx,:]

print (len(dataset.attrs), len(dataset.objs), len(dataset.pairs))
print (X_train.shape, X_test.shape)

if args.generate:
    generate_svms()
    make_svm_tensor()

if args.evalsvm:
    evaluate_svms()

if args.evaltf:
    evaluate_tensorcompletion()
