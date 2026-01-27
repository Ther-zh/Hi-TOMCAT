# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

CURRENT_DIR=$(pwd)


cd data

# download datasets and splits
# wget -c http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip -O utzap.zip
wget -c http://www.cs.cmu.edu/~spurushw/publication/compositional/compositional_split_natural.tar.gz -O compositional_split_natural.tar.gz




# UT-Zappos50k
unzip ut-zap50k-images.zip -d ut-zap50k/
mv ut-zap50k/ut-zap50k-images ut-zap50k/_images/


# Download new splits for Purushwalkam et. al
tar -zxvf splits.tar.gz

cd $CURRENT_DIR
python download_data/reorganize_utzap.py

mv data/ut-zap50k data/ut-zappos
