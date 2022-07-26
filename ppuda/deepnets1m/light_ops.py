# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Light weighted version of common PyTorch modules to speed up frequent creation of networks and training GHNs.
These modules should only be used for training GHNs as their functionality in other cases is not tested.

Example of creating 1000 convolutional layers showing 4x speed up of using Conv2dLight

>>> start = time.time(); len([Conv2d(16,16,1) for i in range(1000)]); time.time() - start
1000
0.08254384994506836
>>> start = time.time(); len([Conv2dLight(16,16,1) for i in range(1000)]); time.time() - start
1000
0.0217587947845459

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from collections import OrderedDict
from typing import Union, Optional, Dict
from torch.nn.modules.conv import _pair
from torch.nn.common_types import _size_2_t


class ModuleLight(nn.Module):
    def __init__(self):
        super(ModuleLight, self).__init__()

    def __setattr__(self, name: str, value: Union[torch.Tensor, 'Module']) -> None:
        if isinstance(value, (list, torch.Tensor, nn.Parameter)) or (name in ['weight', 'bias'] and value is None):
            self._parameters[name] = tuple(value) if isinstance(value, list) else value
        else:
            object.__setattr__(self, name, value)

    def to(self, *args, **kwargs):
        return None

    def reset_parameters(self) -> None:
        return None


class Conv2dLight(ModuleLight):

    def __init__(self, in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=torch.bool):

        super(Conv2dLight, self).__init__()

        self._parameters: Dict[str, Optional[nn.Parameter]] = OrderedDict()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding if isinstance(padding, str) else _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self.weight = [out_channels, in_channels // groups, *self.kernel_size]
        if bias:
            self.bias = [out_channels]
        else:
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class LinearLight(ModuleLight):

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=torch.bool):
        super(LinearLight, self).__init__()
        self._parameters: Dict[str, Optional[nn.Parameter]] = OrderedDict()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = [out_features, in_features]
        if bias:
            self.bias = [out_features]
        else:
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)


class LayerNormLight(ModuleLight):

    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None) -> None:
        super(LayerNormLight, self).__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]

        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        assert elementwise_affine
        self.weight = list(normalized_shape)
        self.bias = list(normalized_shape)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)


class BatchNorm2dLight(ModuleLight):

    def __init__(self, num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=False,
        device=None,
        dtype=None
    ):
        super(BatchNorm2dLight, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        assert affine and not track_running_stats, 'assumed affine and that running stats is not updated'
        self.running_mean = None
        self.running_var = None
        self.num_batches_tracked = None

        self.weight = [num_features]
        self.bias = [num_features]


    def forward(self, input: torch.Tensor) -> torch.Tensor:

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        return F.batch_norm(
            input,
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
