# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
ImageNet wrapper.

"""


import hashlib
from collections import defaultdict
import torchvision


def split_train_and_val(list_of_tups, num_val_per_class=50):

    class_dict = defaultdict(list)

    for item in list_of_tups:
        class_dict[item[1]].append(item)

    train_samples, val_samples = [], []

    # fix the class ordering
    for k in sorted(class_dict.keys()):
        v = class_dict[k]

        # last num_val_per_class will be the val samples
        train_samples.extend(v[:-num_val_per_class])
        val_samples.extend(v[-num_val_per_class:])

    return train_samples, val_samples


class ImageNetDataset(torchvision.datasets.ImageNet):

    def __init__(self, root, split, transform=None, has_validation=True):
        assert split in {"train", "val"}

        # this is to compute the split in case we have validation set, but with
        # the actual validation set being used as the test set, meaning
        # we need to split the train set into train and val
        base_split = 'train' if (split == 'val' and has_validation) else split

        super().__init__(root, base_split, transform=transform)

        # revert the split back to the actual one
        # since pytorch will set the self.split attribute
        self.split = split

        if has_validation:
            train_samples, val_samples = split_train_and_val(
                self.samples, num_val_per_class=50)

            self.samples = train_samples if split == "train" else val_samples

        m = hashlib.sha256()
        m.update(str(self.samples).encode())

        # convert from hex string to number
        self.checksum = int(m.hexdigest(), 16)

    @property
    def num_examples(self):
        return len(self)
