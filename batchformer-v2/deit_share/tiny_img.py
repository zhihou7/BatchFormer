import os
import sys
os.makedirs('/ImageNetTiny')
os.makedirs('/ImageNetTiny/train')
os.makedirs('/ImageNetTiny/val')

with open('temp.txt') as f:
    for item in f.readlines():
        if len(item) < 3:
            continue
        os.system('ln -s /data/train/{} /ImageNetTiny/train/'.format(item.strip()))
        os.system('ln -s /data/val/{} /ImageNetTiny/val/'.format(item.strip()))