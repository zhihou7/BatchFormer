# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Reorganize the UT-Zappos dataset to resemble the MIT-States dataset
root/attr_obj/img1.jpg
root/attr_obj/img2.jpg
root/attr_obj/img3.jpg
...
"""

import os
import torch
import shutil
import tqdm

DATA_FOLDER= "/scratch/ZHIHOU/CZSL"

root = DATA_FOLDER+'/ut-zap50k/'
os.makedirs(root+'/images',exist_ok=True)

data = torch.load(root+'/metadata_compositional-split-natural.t7')
for instance in tqdm.tqdm(data):
	image, attr, obj = instance['_image'], instance['attr'], instance['obj']
	old_file = '%s/_images/%s'%(root, image)
	new_dir = '%s/images/%s_%s/'%(root, attr, obj)
	os.makedirs(new_dir, exist_ok=True)
	shutil.copy(old_file, new_dir)
