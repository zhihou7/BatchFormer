#  Torch imports
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import numpy as np
from flags import DATA_FOLDER

cudnn.benchmark = True

# Python imports
import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj

# Local imports
from data import dataset as dset
from models.common import Evaluator
from utils.utils import load_args
from utils.config_model import configure_model
from flags import parser



device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # Get arguments and start logging
    args = parser.parse_args()
    logpath = args.logpath
    config = [os.path.join(logpath, _) for _ in os.listdir(logpath) if _.endswith('yml')][0]
    load_args(config, args)

    # Get dataset
    trainset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='train',
        split=args.splitname,
        model=args.image_extractor,
        update_features=args.update_features,
        train_only=args.train_only,
        subset=args.subset,
        open_world=args.open_world
    )

    valset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='val',
        split=args.splitname,
        model=args.image_extractor,
        subset=args.subset,
        update_features=args.update_features,
        open_world=args.open_world
    )

    valoader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=8)

    testset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='test',
        split=args.splitname,
        model =args.image_extractor,
        subset=args.subset,
        update_features = args.update_features,
        open_world=args.open_world
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)


    # Get model and optimizer
    image_extractor, model, optimizer = configure_model(args, trainset)
    args.extractor = image_extractor

    if args.add_bt:
        args.load = ospj(logpath,'ckpt_global{}1e-06_best_auc.t7'.format(args.add_bt))
    else:
        args.load = ospj(logpath,'ckpt_best_auc.t7')

    print(args.load)
    checkpoint = torch.load(args.load)
    print(checkpoint['epoch'])
    if image_extractor:
        try:
            image_extractor.load_state_dict(checkpoint['image_extractor'])
            image_extractor.eval()
        except:
            print('No Image extractor in checkpoint')
    net_params = {}
    for k in checkpoint['net'].keys():
        nk = k
        # if k.startswith('gcn.'):
        #     nk = 'model.' + k
        net_params[nk] = checkpoint['net'][k]
    print(net_params.keys())
    model.load_state_dict(net_params, False)
    model.eval()

    threshold = None
    if args.open_world and args.hard_masking:
        assert args.model == 'compcos', args.model + ' does not have hard masking.'
        if args.threshold is not None:
            threshold = args.threshold
        else:
            evaluator_val = Evaluator(valset, model)
            unseen_scores = model.compute_feasibility().to('cpu')
            seen_mask = model.seen_mask.to('cpu')
            min_feasibility = (unseen_scores+seen_mask*10.).min()
            max_feasibility = (unseen_scores-seen_mask*10.).max()
            thresholds = np.linspace(min_feasibility,max_feasibility, num=args.threshold_trials)
            best_auc = 0.
            best_th = -10
            with torch.no_grad():
                for th in thresholds:
                    results = test(image_extractor,model,valoader,evaluator_val,args,threshold=th,print_results=False)
                    auc = results['AUC']
                    if auc > best_auc:
                        best_auc = auc
                        best_th = th
                        print('New best AUC',best_auc)
                        print('Threshold',best_th)

            threshold = best_th

    evaluator = Evaluator(testset, model)

    with torch.no_grad():
        test(image_extractor, model, testloader, evaluator, args, threshold)


def test(image_extractor, model, testloader, evaluator,  args, threshold=None, print_results=True):
        if image_extractor:
            image_extractor.eval()

        model.eval()

        accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []

        for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
            data = [d.to(device) for d in data]

            if image_extractor:
                data[0] = image_extractor(data[0])
            if threshold is None:
                _, predictions = model(data)
            else:
                _, predictions = model.val_forward_with_threshold(data,threshold)

            attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]

            all_pred.append(predictions)
            all_attr_gt.append(attr_truth)
            all_obj_gt.append(obj_truth)
            all_pair_gt.append(pair_truth)

        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt), torch.cat(all_obj_gt), torch.cat(all_pair_gt)

        all_pred_dict = {}
        # Gather values as dict of (attr, obj) as key and list of predictions as values
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k] for i in range(len(all_pred))])

        # Calculate best unseen accuracy
        results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=args.bias, topk=args.topk)
        stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict,
                                               topk=args.topk)


        result = ''
        for key in stats:
            result = result + key + '  ' + str(round(stats[key], 4)) + '| '

        result = result + args.name
        if print_results:
            print(f'Results')
            print(result)
        return results


if __name__ == '__main__':
    main()
