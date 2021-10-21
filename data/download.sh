# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
echo "Downloading DeepNets-1M..."

cd data  # assume we are running this from the root folder

wget https://dl.fbaipublicfiles.com/deepnets1m/deepnets1m_eval.hdf5;    # 16 MB
wget https://dl.fbaipublicfiles.com/deepnets1m/deepnets1m_meta.tar.gz;  # 35 MB
tar -xf deepnets1m_meta.tar.gz;

wget https://dl.fbaipublicfiles.com/deepnets1m/deepnets1m_search.hdf5;  # 1.3 GB
wget https://dl.fbaipublicfiles.com/deepnets1m/deepnets1m_train.hdf5;   # 10.3 GB

cd ..

echo "done!"
