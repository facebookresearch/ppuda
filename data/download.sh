# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
echo "Downloading DeepNets-1M..."

cd data  # assume we are running this from the root folder

wget https://dl.fbaipublicfiles.com/deepnets1m/deepnets1m_eval.hdf5;    # 16 MB (md5: 1f5641329271583ad068f43e1521517e)
wget https://dl.fbaipublicfiles.com/deepnets1m/deepnets1m_meta.tar.gz;  # 35 MB (md5: a42b6f513da6bbe493fc16a30d6d4e3e)
tar -xf deepnets1m_meta.tar.gz;

wget https://dl.fbaipublicfiles.com/deepnets1m/deepnets1m_search.hdf5;  # 1.3 GB (md5: 0a93f4b4e3b729ea71eb383f78ea9b53)
wget https://dl.fbaipublicfiles.com/deepnets1m/deepnets1m_train.hdf5;   # 10.3 GB (md5: 90bbe84bb1da0d76cdc06d5ff84fa23d)

cd ..

echo "done!"
