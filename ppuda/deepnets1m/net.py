# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Some functionality in this script is based on the DARTS code: https://github.com/quark0/darts (Apache License 2.0)

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops import OPS, ReLUConvBN, FactorizedReduce, PosEnc, Zero, Stride, parse_op_ks, bn_layer, lin_layer, conv_layer
from ..utils import drop_path


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C_in, C_out, reduction, reduction_prev,
                 norm='bn', preproc=True, is_vit=False, light=False, cell_ind=0):
        super(Cell, self).__init__()
        self._is_vit = is_vit
        self._cell_ind = cell_ind
        self._has_none = sum([n[0] == 'none' for n in genotype.normal + genotype.reduce]) > 0
        self.genotype = genotype

        if preproc:
            if reduction_prev and not is_vit:
                self.preprocess0 = FactorizedReduce(C_prev_prev, C_out, norm=norm)
            else:
                self.preprocess0 = ReLUConvBN(C_prev_prev, C_out, norm=norm)
            self.preprocess1 = ReLUConvBN(C_prev, C_out, norm=norm)
        else:
            if reduction_prev and not is_vit:
                self.preprocess0 = Stride(stride=2)
            else:
                self.preprocess0 = nn.Identity()
            self.preprocess1 = nn.Identity()

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C_in, C_out, op_names, indices, concat, reduction, norm, light)


    def _compile(self, C_in, C_out, op_names, indices, concat, reduction, norm, light):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for i, (name, index) in enumerate(zip(op_names, indices)):
            stride = 2 if (reduction and index < 2 and not self._is_vit) else 1
            name, ks = parse_op_ks(name)
            self._ops.append(OPS[name](C_in if index <= 1 else C_out, C_out, ks, stride, norm, light))

        self._indices = indices


    def forward(self, s0, s1, drop_path_prob=0):

        s0 = None if (s0 is None or _is_none(self.preprocess0)) else self.preprocess0(s0)
        s1 = None if (s1 is None or _is_none(self.preprocess1)) else self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            s = None

            if not (isinstance(op1, Zero) or _is_none(op1) or h1 is None):
                h1 = op1(h1)
                if self.training and drop_path_prob > 0 and not isinstance(op1, nn.Identity):
                    h1 = drop_path(h1, drop_path_prob)
                s = h1

            if not (isinstance(op2, Zero) or _is_none(op2) or h2 is None):
                h2 = op2(h2)
                if self.training and drop_path_prob > 0 and not isinstance(op2, nn.Identity):
                    h2 = drop_path(h2, drop_path_prob)
                try:
                    s = h2 if s is None else (h1 + h2)
                except:
                    print(h1.shape, h2.shape, self.genotype)
                    raise

            states.append(s)


        if sum([states[i] is None for i in self._concat]) > 0:
            # Replace None states with Zeros to match feature dimensionalities and enable forward pass
            assert self._has_none, self.genotype
            s_dummy = None
            for i in self._concat:
                if states[i] is not None:
                    s_dummy = states[i] * 0
                    break

            if s_dummy is None:
                return None
            else:
                for i in self._concat:
                    if states[i] is None:
                        states[i] = s_dummy

        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes, norm='bn', pool_sz=5, light=False):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(pool_sz, stride=min(pool_sz, 3), padding=0, count_include_pad=False),  # image size = 2 x 2
            conv_layer(light)(C, 128, 1, bias=False),
            bn_layer(norm, 128, light),
            nn.ReLU(inplace=True),
            conv_layer(light)(128, 768, 2, bias=False),
            bn_layer(norm, 768, light),
            nn.ReLU(inplace=True)
        )
        self.classifier = lin_layer(light)(768, num_classes)

    def forward(self, x):
        assert self.training, 'this module is assumed to be used only for training'
        x = self.features(x)
        x = self.classifier(x.reshape(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes, norm='bn', light=False):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            conv_layer(light)(C, 128, 1, bias=False),
            bn_layer(norm, 128, light),
            nn.ReLU(inplace=True),
            conv_layer(light)(128, 768, 2, bias=False),
            # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
            # Commenting it out for consistency with the experiments in the paper.
            # bn_layer(norm, 768),
            nn.ReLU(inplace=True)
        )
        self.classifier = lin_layer(light)(768, num_classes)

    def forward(self, x):
        assert self.training, 'this module is assumed to be used only for training'
        x = self.features(x)
        x = self.classifier(x.reshape(x.size(0), -1))
        return x


class Network(nn.Module):

    def __init__(self,
                 C,
                 num_classes,
                 genotype,
                 n_cells,
                 ks=3,
                 is_imagenet_input=True,
                 stem_pool=False,
                 stem_type=0,
                 is_vit=None,
                 norm='bn-track',
                 preproc=True,
                 C_mult=2,
                 fc_layers=0,
                 fc_dim=0,
                 glob_avg=True,
                 auxiliary=False,
                 light=False
                 ):
        super(Network, self).__init__()

        self.genotype = genotype
        self._auxiliary = auxiliary
        self.drop_path_prob = 0
        self.expected_input_sz = 224 if is_imagenet_input else 32

        self._is_vit = sum([n[0] == 'msa' for n in genotype.normal + genotype.reduce]) > 0 if is_vit is None else is_vit

        steps = len(genotype.normal_concat)  # number of inputs to the concatenation operation
        if steps > 1 or C_mult > 1:
            assert preproc, 'preprocessing layers must be used in this case'

        self._stem_type = stem_type
        assert stem_type in [0, 1], ('either 0 (simple) or 1 (imagenet-style) stem must be chosen', stem_type)

        C_prev_prev = C_prev = C_curr = C

        # Define the stem
        if self._is_vit:
            # Visual Transformer stem
            self.stem0 = OPS['conv_stride'](3, C, 16 if is_imagenet_input else 3, None, norm, light)
            self.pos_enc = PosEnc(C, 14 if is_imagenet_input else 11, light)

        elif stem_type == 0:
            # Simple stem
            C_stem = int(C * (3 if (preproc and not is_imagenet_input) else 1))

            self.stem = nn.Sequential(
                conv_layer(light)(3, C_stem, ks, stride=4 if is_imagenet_input else 1, padding=ks // 2, bias=False),
                bn_layer(norm, C_stem, light),
                nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=False) if stem_pool else nn.Identity(),
            )
            C_prev_prev = C_prev = C_stem

        else:
            # ImageNet-style stem
            self.stem0 = nn.Sequential(
                conv_layer(light)(3, C // 2, kernel_size=ks, stride=2 if is_imagenet_input else 1, padding=ks // 2, bias=False),
                bn_layer(norm, C // 2, light),
                nn.ReLU(inplace=True),
                conv_layer(light)(C // 2, C, kernel_size=3, stride=2 if is_imagenet_input else 1, padding=1, bias=False),
                bn_layer(norm, C, light)
            )

            self.stem1 = nn.Sequential(
                nn.ReLU(inplace=True),
                conv_layer(light)(C, C, 3, stride=2, padding=1, bias=False),
                bn_layer(norm, C, light)
            )

        self._n_cells = n_cells
        self.cells = nn.ModuleList()

        is_reduction = lambda cell_ind: cell_ind in [n_cells // 3, 2 * n_cells // 3] and cell_ind > 0
        self._auxiliary_cell_ind =  2 * n_cells // 3

        reduction_prev = stem_type == 1
        for cell_ind in range(n_cells):
            if is_reduction(cell_ind):
                C_curr *= C_mult
                reduction = True
            else:
                reduction = False

            reduction_next = is_reduction(cell_ind + 1)

            cell = Cell(genotype,
                        C_prev_prev,
                        C_prev,
                        C_in=C_curr if preproc else C_prev,
                        C_out=C_curr * (C_mult if reduction_next and steps == 1 and not preproc else 1),
                        reduction=reduction,
                        reduction_prev=reduction_prev,
                        norm=norm,
                        is_vit=self._is_vit,
                        preproc=preproc,
                        light=light,
                        cell_ind=cell_ind)
            self.cells.append(cell)

            reduction_prev = reduction
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

            if auxiliary and cell_ind == self._auxiliary_cell_ind:
                if is_imagenet_input:
                    self.auxiliary_head = AuxiliaryHeadImageNet(C_prev, num_classes, norm=norm)
                else:
                    self.auxiliary_head = AuxiliaryHeadCIFAR(C_prev, num_classes, norm=norm,
                                                             pool_sz=2 if (stem_type == 1 or stem_pool) else 5)

        self._glob_avg = glob_avg
        if glob_avg:
            self.global_pooling = nn.AdaptiveAvgPool2d(1)
        else:
            if is_imagenet_input:
                s = 7 if (stem_type == 1 or stem_pool) else 14
            else:
                s = 4 if (stem_type == 1 or stem_pool) else 8
            C_prev *= s ** 2

        fc = [lin_layer(light)(C_prev, fc_dim if fc_layers > 1 else num_classes)]
        for i in range(fc_layers - 1):
            assert fc_dim > 0, fc_dim
            fc.append(nn.ReLU(inplace=True))
            fc.append(nn.Dropout(p=0.5, inplace=False))
            fc.append(lin_layer(light)(in_features=fc_dim, out_features=fc_dim if i < fc_layers - 2 else num_classes))
        self.classifier = nn.Sequential(*fc)

        if light:
            self.__dict__['_layered_modules'] = named_layered_modules(self)


    def forward(self, input):

        if self._is_vit:
            s0 = self.stem0(input)
            s0 = s1 = self.pos_enc(s0)
        else:
            if self._stem_type == 1:
                s0 = self.stem0(input)
                s1 = None if _is_none(self.stem1) else self.stem1(s0)
            else:
                s0 = s1 = self.stem(input)

        logits_aux = None
        for cell_ind, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if self._auxiliary and cell_ind == self._auxiliary_cell_ind and self.training:
                logits_aux = self.auxiliary_head(F.adaptive_avg_pool2d(s1, 8)
                                                 if self._is_vit and self.expected_input_sz == 32
                                                 else s1)
        if s1 is None:
            raise ValueError('the network has invalid configuration: the output is None')
        out = self.global_pooling(s1) if self._glob_avg else s1
        logits = self.classifier(out.reshape(out.size(0), -1))

        return logits, logits_aux


"""
Helper functions to train GHNs and work with DeepNets-1M.
"""

def _is_none(mod):
    for n, p in mod.named_modules():
        if hasattr(p, 'weight') and p.weight is None:
            return True
    return False


def get_cell_ind(param_name, layers=1):
    if param_name.find('cells.') >= 0:
        pos1 = len('cells.')
        pos2 = pos1 + param_name[pos1:].find('.')
        cell_ind = int(param_name[pos1: pos2])
    elif param_name.startswith('classifier') or param_name.startswith('auxiliary'):
        cell_ind = layers - 1
    elif layers == 1 or param_name.startswith('stem') or param_name.startswith('pos_enc'):
        cell_ind = 0
    else:
        cell_ind = None

    return cell_ind


def named_layered_modules(model):
    if hasattr(model, 'module'):  # in case of multigpu model
        model = model.module
    layers = model._n_cells if hasattr(model, '_n_cells') else 1
    layered_modules = [{} for _ in range(layers)]
    cell_ind = 0
    for module_name, m in model.named_modules():

        cell_ind = m._cell_ind if hasattr(m, '_cell_ind') else cell_ind

        is_layer_scale = hasattr(m, 'layer_scale') and m.layer_scale is not None    # support of PyTorch ConvNext
        is_proj_w = hasattr(m, 'in_proj_weight') and m.in_proj_weight is not None   # support of PyTorch ViT/Swin
        is_pos_enc = hasattr(m, 'pos_embedding') and m.pos_embedding is not None    # support of PyTorch ViT/Swin
        is_w = (hasattr(m, 'weight') and m.weight is not None) or is_proj_w or is_pos_enc or is_layer_scale
        is_proj_b = hasattr(m, 'in_proj_bias') and m.in_proj_bias is not None       # support of PyTorch ViT/Swin
        is_b = (hasattr(m, 'bias') and m.bias is not None) or is_proj_b

        if is_w or is_b:
            if module_name.startswith('module.'):
                module_name = module_name[module_name.find('.') + 1:]

            if is_w:
                key = module_name + ('.layer_scale' if is_layer_scale else ('.in_proj_weight'
                                                                            if is_proj_w else
                                                                            ('.pos_embedding.weight'
                                                                             if is_pos_enc else '.weight')))
                w = m.layer_scale if is_layer_scale else (m.in_proj_weight if is_proj_w else
                                                          (m.pos_embedding if is_pos_enc else m.weight))
                layered_modules[cell_ind][key] = {'param_name': key, 'module': m, 'is_w': True,
                                                  'sz': tuple(w) if isinstance(w, (list, tuple)) else w.shape}
            if is_b:
                key = module_name + ('.in_proj_bias' if is_proj_b else '.bias')
                b = m.in_proj_bias if is_proj_b else m.bias
                layered_modules[cell_ind][key] = {'param_name': key, 'module': m, 'is_w': False,
                                                  'sz': tuple(b) if isinstance(b, (list, tuple)) else b.shape}

    return layered_modules
