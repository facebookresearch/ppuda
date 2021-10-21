# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Example of predicting parameters for the Visual Transformer-based architecture from
"Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,
Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly,
Jakob Uszkoreit, Neil Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021."
(https://arxiv.org/abs/2010.11929)

The script predicts parameters for the network and evaluates them on CIFAR-10 or ImageNet.

Example:

    python examples/vit.py imagenet

"""


import sys
from ppuda.vision.loader import image_loader
from ppuda.ghn.nn import GHN2
from ppuda.deepnets1m.genotypes import ViT
from ppuda.deepnets1m.net import Network
from ppuda.utils import capacity, infer


try:
    dataset = sys.argv[1].lower()   # imagenet, cifar10
    ghn = GHN2(dataset)
except:
    print('\nExample of usage: python examples/vit.py imagenet\n')
    raise

is_imagenet = dataset == 'imagenet'
images_val, num_classes = image_loader(dataset, num_workers=8 * is_imagenet)[1:]
if is_imagenet:
    images_val.sampler.generator.manual_seed(1111)  # set the generator seed to reproduce results

# Define the network configuration
model = Network(C=128,
                num_classes=num_classes,
                genotype=ViT,
                n_cells=12,
                preproc=False,
                C_mult=1,
                is_imagenet_input=is_imagenet).eval()


model = ghn(model)  # Predict all parameters for ViT

print('\nEvaluation of ViT with {} parameters'.format(capacity(model)[1]))

top1, top5 = infer(model, images_val, verbose=True)
# top5=4.41 for ViT on ImageNet and top1=11.41 on CIFAR-10
if (is_imagenet and abs(top5 - 4.41) > 0.01) or (not is_imagenet and top1 != 11.41):
    print('WARNING: results appear to be different from expected!' )

print('\ndone')
