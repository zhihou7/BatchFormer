# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

CURRENT_DIR=$(pwd)

FOLDER=$1

mkdir $FOLDER
mkdir $FOLDER/fast
mkdir $FOLDER/glove
mkdir $FOLDER/w2v

wget --show-progress -O utils/cgqa-graph.t7 https://raw.githubusercontent.com/ExplainableML/czsl/main/utils/cgqa-graph.t7
wget --show-progress -O utils/mitstates-graph.t7 https://raw.githubusercontent.com/ExplainableML/czsl/main/utils/mitstates-graph.t7
wget --show-progress -O utils/utzappos-graph.t7 https://raw.githubusercontent.com/ExplainableML/czsl/main/utils/utzappos-graph.t7

cp utils/download_embeddings.py $FOLDER/fast


# Download everything
wget --show-progress -O $FOLDER/attr-ops-data.tar.gz https://www.cs.utexas.edu/~tushar/attribute-ops/attr-ops-data.tar.gz
wget --show-progress -O $FOLDER/mitstates.zip http://wednesday.csail.mit.edu/joseph_result/state_and_transformation/release_dataset.zip
wget --show-progress -O $FOLDER/utzap.zip http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip
wget --show-progress -O $FOLDER/splits.tar.gz https://www.senthilpurushwalkam.com/publication/compositional/compositional_split_natural.tar.gz
wget --show-progress -O $FOLDER/cgqa.zip https://s3.mlcloud.uni-tuebingen.de/czsl/cgqa.zip

echo "Data downloaded. Extracting files..."
sed -i "s|ROOT_FOLDER|$FOLDER|g" utils/reorganize_utzap.py
sed -i "s|ROOT_FOLDER|$FOLDER|g" flags.py

# Dataset metadata, pretrained SVMs and features, tensor completion data

cd $FOLDER

tar -zxvf attr-ops-data.tar.gz --strip 1

# MIT-States
unzip mitstates.zip 'release_dataset/images/*' -d mit-states/
mv mit-states/release_dataset/images mit-states/images/
rm -r mit-states/release_dataset
rename "s/ /_/g" mit-states/images/*

# UT-Zappos50k
unzip utzap.zip -d ut-zap50k/
mv ut-zap50k/ut-zap50k-images ut-zap50k/_images/

# C-GQA
unzip cgqa.zip -d cgqa/

# Download new splits for Purushwalkam et. al
tar -zxvf splits.tar.gz

# remove all zip files and temporary files
rm -r attr-ops-data.tar.gz mitstates.zip utzap.zip splits.tar.gz cgqa.zip

# Download embeddings

# Glove (from attribute as operators)
mv data/glove/* glove/

# FastText
cd fast
python download_embeddings.py
rm cc.en.300.bin.gz

# Word2Vec
cd ../w2v
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
gzip -d GoogleNews-vectors-negative300.bin.gz
rm GoogleNews-vectors-negative300.bin.gz

cd ..
rm -r data

cd $CURRENT_DIR
python utils/reorganize_utzap.py
