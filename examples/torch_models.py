# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Example of predicting parameters for the networks from torchvision, such as ResNet-50.
See possible networks at https://pytorch.org/vision/stable/models.html.
The script predicts parameters for the networks and evaluates them on CIFAR-10 or ImageNet.

Example:

    python examples/torch_models.py imagenet resnet50

"""


import torchvision
import sys
from ppuda.vision.loader import image_loader
from ppuda.ghn.nn import GHN2
from ppuda.utils import capacity, adjust_net, infer


try:
    dataset = sys.argv[1].lower()  # imagenet, cifar10
    arch = sys.argv[2].lower()  # resnet50, wide_resnet101_2, etc.

    ghn = GHN2(dataset)
except:
    print('\nExample of usage: python examples/torch_models.py imagenet resnet50\n')
    raise


is_imagenet = dataset == 'imagenet'
images_val, num_classes = image_loader(dataset, num_workers=8 * is_imagenet)[1:]
if is_imagenet:
    images_val.sampler.generator.manual_seed(1111)  # set the generator seed to reproduce results

kw_args = {'aux_logits': False, 'init_weights': False} if arch == 'googlenet' else {}  # ignore auxiliary outputs in googlenet for this example

# Predict all parameters (any network from torchvision.models can be used)
model = ghn(adjust_net(eval('torchvision.models.%s(num_classes=%d, **kw_args)' % (arch, num_classes)),
                       large_input=is_imagenet))

print('\nEvaluation of {} with {} parameters'.format(arch.upper(), capacity(model)[1]))

top1, top5 = infer(model, images_val, verbose=True)
# top5=5.27 for ResNet-50 on ImageNet and top1=58.62 on CIFAR-10
if arch == 'resnet50':
    if (is_imagenet and abs(top5 - 5.27) > 0.01) or (not is_imagenet and top1 != 58.62):
        print('WARNING: results appear to be different from expected!' )

print('\ndone')
