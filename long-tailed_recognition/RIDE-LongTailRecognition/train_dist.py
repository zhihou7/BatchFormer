import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
import torch.distributed as dist
import torch.utils.data.distributed
import os

# For num_experts with same settings, we don't want this to set to True.
# This is strongly discouraged because it's misleading: setting it to true does not make it reproducible acorss machine/pytorch version. In addition, it also makes training slower. Use with caution.
deterministic = False
if deterministic:
    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

def main(config, args):
    print(args)
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    ngpus_per_node = torch.cuda.device_count()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node
        print(args.rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    print(config['name'])
    import os
    os.environ['DATASET_N'] = config['name']
    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    # if args.distributed:
    #     model.cuda()
    #     # DistributedDataParallel will divide and allocate batch_size to all
    #     # available GPUs if device_ids are not set
    #     model = torch.nn.parallel.DistributedDataParallel(model)

    # get function handles of loss and metrics
    loss_class = getattr(module_loss, config["loss"]["type"])
    if hasattr(loss_class, "require_num_experts") and loss_class.require_num_experts:
        criterion = config.init_obj('loss', module_loss, cls_num_list=data_loader.cls_num_list, num_experts=config["arch"]["args"]["num_experts"])
    else:
        criterion = config.init_obj('loss', module_loss, cls_num_list=data_loader.cls_num_list)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer = config.init_obj('optimizer', torch.optim, model.parameters())

    if "type" in config._config["lr_scheduler"]:
        if config["lr_scheduler"]["type"] == "CustomLR":
            lr_scheduler_args = config["lr_scheduler"]["args"]
            gamma = lr_scheduler_args["gamma"] if "gamma" in lr_scheduler_args else 0.1
            print("Scheduler step1, step2, warmup_epoch, gamma:", (lr_scheduler_args["step1"], lr_scheduler_args["step2"], lr_scheduler_args["warmup_epoch"], gamma))
            def lr_lambda(epoch):
                if epoch >= lr_scheduler_args["step2"]:
                    lr = gamma * gamma
                elif epoch >= lr_scheduler_args["step1"]:
                    lr = gamma
                else:
                    lr = 1

                """Warmup"""
                warmup_epoch = lr_scheduler_args["warmup_epoch"]
                if epoch < warmup_epoch:
                    lr = lr * float(1 + epoch) / warmup_epoch

                print('lr', epoch, lr)
                return lr
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif config["lr_scheduler"]["type"] == "CosLR":
            lr_scheduler_args = config["lr_scheduler"]["args"]
            all_epochs = config['trainer']['epochs']
            def lr_lambda1(epoch):
                import math
                warmup_epoch = lr_scheduler_args["warmup_epoch"]
                lr = config['optimizer']['args']['lr']
                if epoch < warmup_epoch:
                    lr = lr * float(1 + epoch) / warmup_epoch
                else:
                    lr = 1.
                    lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epoch + 1 ) / (all_epochs - warmup_epoch + 1 )))
                print('lr', epoch, lr)
                print([group['initial_lr'] for group in optimizer.param_groups])
                return lr
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda1)
        else:
            lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    else:
        lr_scheduler = None

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler, distributed=args.distributed,
                      pretrained=args.pretrained, rank=args.rank,
                      add_bt = args.add_bt)

    trainer.train_dist(args, ngpus_per_node)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    args.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    args.add_argument('--dist-url', dest='dist_url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    args.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    args.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    args.add_argument('--pretrained', default=None, type=str,
                      help='use a pretrained model')
    args.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--name'], type=str, target='name'),
        CustomArgs(['--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--step1'], type=int, target='lr_scheduler;args;step1'),
        CustomArgs(['--step2'], type=int, target='lr_scheduler;args;step2'),
        CustomArgs(['--warmup'], type=int, target='lr_scheduler;args;warmup_epoch'),
        CustomArgs(['--gamma'], type=float, target='lr_scheduler;args;gamma'),
        CustomArgs(['--save_period'], type=int, target='trainer;save_period'),
        CustomArgs(['--reduce_dimension'], type=int, target='arch;args;reduce_dimension'),
        CustomArgs(['--layer2_dimension'], type=int, target='arch;args;layer2_output_dim'),
        CustomArgs(['--layer3_dimension'], type=int, target='arch;args;layer3_output_dim'),
        CustomArgs(['--layer4_dimension'], type=int, target='arch;args;layer4_output_dim'),
        CustomArgs(['--num_experts'], type=int, target='arch;args;num_experts'),
        CustomArgs(['--distribution_aware_diversity_factor'], type=float, target='loss;args;additional_diversity_factor'),
        CustomArgs(['--pos_weight'], type=float, target='arch;args;pos_weight'),
        CustomArgs(['--collaborative_loss'], type=int, target='loss;args;collaborative_loss'),
        CustomArgs(['--distill_checkpoint'], type=str, target='distill_checkpoint'),
        CustomArgs(['--add_bt'], type=int, target='arch;args;add_bt'),
        CustomArgs(['--add_bt_teacher'], type=int, target='distill_arch;args;add_bt')
    ]
    config = ConfigParser.from_args(args, options)
    main(config, args.parse_args())
