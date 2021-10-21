# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Example of predicting parameters for all the networks from torchvision, such as ResNet-50, VGG, MobileNet, etc.
See the list of networks at https://pytorch.org/vision/stable/models.html.
The script predicts parameters for all the networks and evaluates them on CIFAR-10 or ImageNet.

Example:

    python examples/all_torch_models.py imagenet

"""


import torchvision
import sys
from ppuda.vision.loader import image_loader
from ppuda.ghn.nn import GHN2
from ppuda.utils import capacity, adjust_net, infer


try:
    dataset = sys.argv[1].lower()  # imagenet, cifar10
    is_imagenet = dataset == 'imagenet'
    ghn = GHN2(dataset)
except:
    print('\nExample of usage: python examples/all_torch_models.py cifar10\n')
    raise


# List all image classification networks from torchvision.models
# skip inception_v3 in this example, since it expects a larger input, so additional image transforms are required
images_val = None
for arch in (
        torchvision.models.resnet.__all__ + ['alexnet'] + torchvision.models.vgg.__all__ +
        torchvision.models.squeezenet.__all__ + torchvision.models.densenet.__all__ + ['googlenet'] +
        torchvision.models.mobilenet.__all__ + torchvision.models.mnasnet.__all__ + torchvision.models.shufflenetv2.__all__):

    if arch[0].isupper():
        continue  # classname

    if is_imagenet or images_val is None:
        images_val, num_classes = image_loader(dataset, num_workers=8 * is_imagenet)[1:]  # reload imagenet val to enable reproducibility

    kw_args = {'aux_logits': False, 'init_weights': False} if arch == 'googlenet' else {}

    # Predict all parameters
    model = ghn(adjust_net(eval('torchvision.models.%s(num_classes=%d, **kw_args)' % (arch, num_classes)),
                           large_input=is_imagenet))

    print('\nEvaluation of {} with {} parameters'.format(arch.upper(), capacity(model)[1]))
    top1, top5 = infer(model, images_val, verbose=True)

print('\ndone')
