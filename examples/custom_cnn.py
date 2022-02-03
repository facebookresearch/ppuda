# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Example of predicting parameters for a user-defined architecture.

The script predicts parameters for the network and evaluates them on CIFAR-10.

Example:

    python examples/custom_cnn.py

"""


import torch.nn as nn
from ppuda.vision.loader import image_loader
from ppuda.ghn.nn import GHN2
from ppuda.utils import capacity, infer

dataset = 'cifar10'
ghn = GHN2(dataset)

is_imagenet = dataset == 'imagenet'
images_val, num_classes = image_loader(dataset, num_workers=8 * is_imagenet)[1:]
if is_imagenet:
    images_val.sampler.generator.manual_seed(1111)  # set the generator seed to reproduce results


# Define the network configuration
class CNN(nn.Module):
    def __init__(self, C=64):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, C, kernel_size=5),
                                    nn.AvgPool2d(2),
                                    nn.ReLU(),
                                    nn.Conv2d(C, C * 2, kernel_size=3),
                                    nn.AvgPool2d(2),
                                    nn.ReLU(),
                                    nn.Conv2d(C * 2, C * 4, kernel_size=3),
                                    nn.ReLU(),
                                    nn.AdaptiveAvgPool2d(1),
                                    nn.Flatten(),
                                    nn.Linear(C * 4, num_classes),
                                    )

    def forward(self, x):
        return self.layers(x)


model = CNN().eval()    # Create the net
model.expected_input_sz = 224 if is_imagenet else 32  # to construct the graph
model = ghn(model)      # Predict all parameters for the model

print('\nEvaluation of CNN with {} parameters'.format(capacity(model)[1]))

top1, top5 = infer(model, images_val, verbose=True)
# top1=16.84 for this CNN on CIFAR-10
if top1 != 16.84:
    print('WARNING: results appear to be different from expected!' )

print('\ndone')
