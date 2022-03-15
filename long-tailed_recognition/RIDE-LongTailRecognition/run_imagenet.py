#!/usr/local/anaconda/4.2.0/bin/python



import subprocess
from argparse import ArgumentParser

import numpy as np
import re

import time
import json
import os

HPC_NODE_INFO_ALLOC_DT = []
HOST_IP = {}
for host in HPC_NODE_INFO_ALLOC_DT:
    ip = host.replace('hpc', '')
    ip = '192.168.66.'+ip
    HOST_IP[host+'.xxx'] = ip
print(HOST_IP)
REMAINDER = '...'

def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="PyTorch distributed training launch "
                                        "helper utilty that will spawn up "
                                        "multiple distributed processes")

    # Optional arguments for the launch helper
    parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training")

    # positional
    parser.add_argument("training_script", type=str, default='./configs/r50_deformable_detr.sh',
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()



if __name__ == "__main__":
    import sys
    import os
    args = parse_args()
    os.system('chmod a+x {}'.format(args.training_script))

    fpbs  = open(os.environ['PBS_NODEFILE'])
    hosts = [item.strip() for item in fpbs.readlines()]
    uniq_hosts = set(hosts)
    host_gpus = {k:0 for k in uniq_hosts}
    host_rank = {k:0 for k in uniq_hosts}
    current_gpus = {}
    for host in hosts:
        host_gpus[host] += args.nproc_per_node
            # host_rank[host] += sum()
        current_gpus[host] = 0
    print(uniq_hosts, hosts)
    uniq_hosts = hosts

    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = HOST_IP[hosts[0]]
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(args.nnodes*args.nproc_per_node)
    current_env["GPUS_PER_NODE"] = str(args.nproc_per_node)
    current_env["NNODES"] = str(len(uniq_hosts))
    processes = []

    alloc_procs = 0
    tmp_i = 0

    current_gpus[hosts[0]] += args.nproc_per_node
    for ni in range(len(uniq_hosts)-1, -1, -1):
        # each process's rank
        current_env["GPUS_PER_NODE"] = str(args.nproc_per_node)
        command_remote = '{} --dist-url tcp://{}:{} --world-size {} --rank {} --dist-backend nccl --multiprocessing-distributed {}'.format(
            args.training_script, HOST_IP[hosts[0]], str(args.master_port), args.nnodes, ni,
            " ".join(args.training_script_args))
        current_env["NODE_RANK"] = str(ni)
        if args.nproc_per_node == 1:
            current_env["CUDA_VISIBLE_DEVICES"] = "0"
        elif args.nproc_per_node == 2:
            current_env["CUDA_VISIBLE_DEVICES"] = "0,1"
        else:
            current_env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        command = '{} {}'.format(str(args.nnodes*args.nproc_per_node), args.training_script, "".join(args.training_script_args))
        command = command_remote
        # python train_img.py --dist-url tcp://192.168.66.239:53212 --dist-backend nccl --multiprocessing-distributed --world-size 2 --rank 1 -b 512 -a resnet18 data/ImageNet_LT/ --add_bt true
        if ni == 0:
            print('cur===========', ni, command)
            # print(host_gpus)
            # print(current_env["GPUS_PER_NODE"], args.nproc_per_node, 'test')
            process = subprocess.Popen(command.split(' '), env=current_env)
        else:
            # print('ssh', ni)
            cmd = 'ssh {} cd {} & {}'.format(hosts[ni], os.path.abspath('.'), command)
            cmd = "ssh {} 'cd {} ; {}'".format(hosts[ni], os.path.abspath('.'), command_remote)
            # print(cmd)
            # os.system(cmd)
            # continue

            if host_gpus[hosts[ni]] == 4 and args.nproc_per_node == 2:
                tmp_i += 1
                current_gpus[hosts[ni]] += args.nproc_per_node
                if current_gpus[hosts[ni]] == 4:
                    arr = ['ssh', hosts[ni], 'cd {} ;CUDA_VISIBLE_DEVICES=2,3 {} '.format(os.path.abspath('.'), command_remote)]
                else:
                    arr = ['ssh', hosts[ni], 'cd {} ;CUDA_VISIBLE_DEVICES=0,1 {} '.format(os.path.abspath('.'), command_remote)]
            elif host_gpus[hosts[ni]] > args.nproc_per_node:
                arr = ['ssh', hosts[ni],
                     'cd {} ;CUDA_VISIBLE_DEVICES={} {} '.format(os.path.abspath('.'), current_gpus[hosts[ni]], command_remote)]

                current_gpus[hosts[ni]] += args.nproc_per_node
            else:
                arr = ['ssh', hosts[ni], 'cd {} ; {} '.format(os.path.abspath('.'), command_remote)]
            print(ni, '============',' '.join(arr))
            process = subprocess.Popen(arr, env=current_env)
        processes.append(process)
        time.sleep(3)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=process.args)
