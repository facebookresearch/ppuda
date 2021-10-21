# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Some functionality in this script is based on the DARTS code: https://github.com/quark0/darts (Apache License 2.0)

"""


import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms


class Cutout:
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class Noise:
    def __init__(self, noise_std=0.08):
        """
        Add Gaussian noise to images.
        :param noise_std: default is 0.08 for CIFAR-10 and 0.18 for ImageNet.
                          The values are according to medium noise severity from https://github.com/hendrycks/robustness.
                          "Dan Hendrycks, Thomas Dietterich. Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. ICLR 2019."
        """
        self.noise_std = noise_std
        print('evaluate on noisy images with std={:.2f}'.format(noise_std))

    def __call__(self, img):
        img = torch.clip(img + torch.randn(size=img.shape) * self.noise_std, 0, 1)
        return img


def transforms_cifar(cutout=False, cutout_length=None, noise=False, sz=32):
    train_transform = [ transforms.RandomCrop(32, padding=4) ]
    valid_transform = []
    if sz == 32:
        normalize = transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124],
                                         std=[0.24703233, 0.24348505, 0.26158768])
    else:
        # Assume the model was trained on ImageNet and is fine-tuned on CIFAR
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_transform.append(transforms.Resize((sz, sz)))
        valid_transform.append(transforms.Resize((sz, sz)))

    train_transform.extend([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    if cutout:
        train_transform.append(Cutout(cutout_length))
    train_transform = transforms.Compose(train_transform)

    valid_transform.append(transforms.ToTensor())
    if noise:
        valid_transform.append(Noise(0.08))
    valid_transform.append(normalize)
    valid_transform = transforms.Compose(valid_transform)

    return train_transform, valid_transform


def transforms_imagenet(noise=False, cifar_style=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = [
        transforms.RandomResizedCrop((32, 32) if cifar_style else 224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2),
        transforms.ToTensor(),
        normalize,
    ]
    if cifar_style:
        del train_transform[2]
    train_transform = transforms.Compose(train_transform)

    valid_transform = [
        transforms.Resize((32, 32) if cifar_style else 256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]
    if cifar_style:
        del valid_transform[1]
    if noise:
        valid_transform.append(Noise(0.08 if cifar_style else 0.18))
    valid_transform.append(normalize)
    valid_transform = transforms.Compose(valid_transform)

    return train_transform, valid_transform
