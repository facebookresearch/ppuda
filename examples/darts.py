# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Example of predicting parameters for DARTS-based architecture from
"Hanxiao Liu, Karen Simonyan, Yiming Yang. DARTS: Differentiable Architecture Search. ICLR 2019."
(https://arxiv.org/abs/1806.09055)

The script predicts parameters for the network and evaluates them on CIFAR-10 or ImageNet.

Example:

    python examples/darts.py imagenet

"""


import sys
from ppuda.vision.loader import image_loader
from ppuda.ghn.nn import GHN2
from ppuda.deepnets1m.genotypes import DARTS
from ppuda.deepnets1m.net import Network
from ppuda.utils import capacity, infer

try:
    dataset = sys.argv[1].lower()   # imagenet, cifar10
    ghn = GHN2(dataset)
except:
    print('\nExample of usage: python examples/darts.py imagenet\n')
    raise

is_imagenet = dataset == 'imagenet'
images_val, num_classes = image_loader(dataset, num_workers=8 * is_imagenet)[1:]
if is_imagenet:
    images_val.sampler.generator.manual_seed(1111)  # set the generator seed to reproduce results

# Define the network configuration based on https://github.com/quark0/darts
model = Network(C=48 if is_imagenet else 36,
                num_classes=num_classes,
                genotype=DARTS,
                n_cells=14 if is_imagenet else 20,
                norm='bn-track',
                stem_type=int(is_imagenet),
                is_imagenet_input=is_imagenet).eval()


model = ghn(model)  # Predict all parameters for DARTS

# To eval the DARTS model trained on ImageNet
# model = load_DARTS_pretrained(model.eval())[0].to('cuda')

print('\nEvaluation of DARTS with {} parameters'.format(capacity(model)[1]))

top1, top5 = infer(model, images_val, verbose=True)
# top5=12.82 for DARTS on ImageNet and top1=24.24 on CIFAR-10
if (is_imagenet and abs(top5 - 12.82) > 0.01) or (not is_imagenet and top1 != 24.24):
    print('WARNING: results appear to be different from expected!' )

print('\ndone')
