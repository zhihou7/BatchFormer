#!/usr/local/anaconda/4.2.0/bin/python

##################################
'''
Code is based on hpc_check_py36_withquotawarning_pbs.py
You can put this file into ~/bin/ and rename filename to ssub (without suffix .py).
Meanwhile chmod a+x ~/bin/ssub
Then, you can submit job by ssub xxx.pbs
This script will search suitable node to submit job with host=xxxx
'''
#################################


import subprocess
from argparse import ArgumentParser

import numpy as np
import re

import time
import json
import os

MEM_PER_NODE = 187
CPU_PER_NODE = 36
GPU_PER_NODE = 4

# from user_quota_config import *
HPC_NODE_INFO_ALLOC_DT = [ 'hpc223', 'hpc224', 'hpc225', 'hpc226', 'hpc227', 'hpc228', 'hpc229', 'hpc230', 'hpc231',
                          'hpc232',
                          'hpc233', 'hpc234', 'hpc235', 'hpc236', 'hpc237', 'hpc238', 'hpc239', 'hpc246', 'hpc247',
                          'hpc248']
HPC_NODE_INFO_ALLOC_DT = ['hpc'+str(i) for i in range(216, 223, 1)] + HPC_NODE_INFO_ALLOC_DT

print(HPC_NODE_INFO_ALLOC_DT )
HOST_IP = {}
for host in HPC_NODE_INFO_ALLOC_DT:
    ip = host.replace('hpc', '')
    ip = '192.168.66.'+ip
    HOST_IP[host+'.hpc.sydney.edu.au'] = ip
print(HOST_IP)

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
    # parser.add_argument('training_script_args', nargs=REMAINDER)
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
    pbs_jobid = os.environ['PBS_JOBID']
    pbs_jobid = pbs_jobid.split('.')[0]

    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = HOST_IP[hosts[0]]
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(args.nnodes*args.nproc_per_node)
    current_env["GPUS_PER_NODE"] = str(args.nproc_per_node)
    current_env["NNODES"] = str(len(uniq_hosts))
    processes = []

    alloc_procs = 0
    tmp_i = 0
    for ni in range(len(uniq_hosts)-1, -1, -1):
        # each process's rank
        current_env["GPUS_PER_NODE"] = str(args.nproc_per_node)
        command_remote = 'NNODES={} MASTER_ADDR={} MASTER_PORT={} NODE_RANK={} GPUS_PER_NODE={} ./tools/run_dist_launch.sh {} {}'.format(
            args.nnodes, HOST_IP[hosts[0]], str(args.master_port), ni, args.nproc_per_node, str(args.nnodes*args.nproc_per_node), args.training_script)
        current_env["NODE_RANK"] = str(ni)
        command = './tools/run_dist_launch.sh {} {}'.format(str(args.nnodes*args.nproc_per_node), args.training_script)

        if ni == 0:
            print('cur', ni, command)
            # print(host_gpus)
            # print(current_env["GPUS_PER_NODE"], args.nproc_per_node, 'test')
            process = subprocess.Popen(command.split(' '), env=current_env)
            # processes.append(process)
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
            print(' '.join(arr))
            process = subprocess.Popen(arr,
                                       env=current_env)
        processes.append(process)
        time.sleep(3)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=process.args)
