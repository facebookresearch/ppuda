# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Trains/fine-tunes a neural network with SGD on the Penn-Fudan object detection task.
The script is based on this tutorial: http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html.

Example:

    Fine-tune DARTS initialized with the parameters predicted by GHN-2-ImageNet:
    python experiments/sgd/detector/train_detector.py --arch DARTS --ckpt ./checkpoints/ghn2_imagenet.pt --init_channels 48 --layers 14

"""


import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator

import os
from ppuda.config import init_config
from ppuda.utils import pretrained_model, capacity
from ppuda.deepnets1m.net import Network
from ppuda.deepnets1m.loader import DeepNets1M
from ppuda.deepnets1m.genotypes import ViT, DARTS
import ppuda.deepnets1m.genotypes as genotypes
from ppuda.ghn.nn import GHN
from penn import PennFudanDataset
from engine import train_one_epoch, evaluate
import utils as utils
import transforms as T


def main():

    args = init_config(mode='train_net')

    # Penn-Fudan dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = PennFudanDataset(os.path.join(args.data_dir, args.dataset),
                               transforms=T.Compose([T.ToTensor(), T.RandomHorizontalFlip(0.5)]))
    dataset_test = PennFudanDataset(os.path.join(args.data_dir, args.dataset), transforms=T.ToTensor())

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_detection(args, num_classes).to(args.device)

    # construct an optimizer
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(args.epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, args.device, epoch, print_freq=args.log_interval)
        # update the learning rate
        lr_scheduler.step()

        if args.save:
            checkpoint_path = os.path.join(args.save, 'checkpoint.pt')
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, checkpoint_path)
            print('\nsaved the checkpoint to {}'.format(checkpoint_path))

        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=args.device)
        torch.cuda.empty_cache()  # trying to avoid occasional issues with GPU memory

    print("That's it!")


def get_backbone(args):

    try:
        genotype = eval('genotypes.%s' % args.arch)
        net_args = {'C': args.init_channels,  # 48 if genotype == DARTS else 128
                    'genotype': genotype,
                    'n_cells': args.layers,   # 14 if genotype == DARTS else 12
                    'C_mult': int(genotype != ViT) + 1,  # assume either ViT or DARTS-style architecture
                    'preproc': genotype != ViT,
                    'stem_type': 1}  # assume that the ImageNet-style stem is used by default
    except:
        deepnets = DeepNets1M(split=args.split,
                              nets_dir=args.data_dir,
                              large_images=True,
                              arch=args.arch)
        assert len(deepnets) == 1, 'one architecture must be chosen to train'
        graph = deepnets[0]
        net_args, idx = graph.net_args, graph.net_idx
        if 'norm' in net_args and net_args['norm'] == 'bn':
            net_args['norm'] = 'bn-track'
        if net_args['genotype'] == ViT:
            net_args['stem_type'] = 1  # using ImageNet style stem even for ViT

    num_classes = 1000
    if isinstance(net_args['genotype'], str):
        model = eval('torchvision.models.%s(pretrained=%d)' % (net_args['genotype'], args.pretrained))
        model.out_channels = model.fc.in_features
    else:
        model = Network(num_classes=num_classes,
                        is_imagenet_input=True,
                        is_vit=False,
                        **net_args)
        model.out_channels = net_args['C'] * len(net_args['genotype'].normal_concat) * (net_args['C_mult'] ** 2)

    if args.ckpt is not None or isinstance(model, torchvision.models.ResNet):
        model = pretrained_model(model, args.ckpt, num_classes, 1, GHN)

    # Allow the detector to use this backbone just as a feature extractor without modifying backbone's code
    def fw_hook(module, input, output):
        if isinstance(input, tuple):
            input = input[0]
        if isinstance(output, tuple):
            output = output[0]
        module.input_sz = input.shape
        if hasattr(module, 'prev_mod') and hasattr(module.prev_mod, 'input_sz'):
            output = output.view(module.prev_mod.input_sz)
        return output

    def add_fw_hooks(m):
        m.register_forward_hook(fw_hook)

    if isinstance(net_args['genotype'], str):
        model.fc = nn.Identity()
        model.avgpool = nn.Identity()
        model.fc.prev_mod = model.avgpool
    else:
        model.classifier = nn.Identity()
        model.global_pooling = nn.Identity()
        model.classifier.prev_mod = model.global_pooling

    model.apply(add_fw_hooks)

    return model


def get_model_detection(args, num_classes):

    # load a pre-trained model for classification and return
    # only the features
    if args.arch not in ['', None]:
        backbone = get_backbone(args)

        # let's make the RPN generate 5 x 3 anchors per spatial
        # location, with 5 different sizes and 3 different aspect
        # ratios. We have a Tuple[Tuple[int]] because each feature
        # map could potentially have different sizes and
        # aspect ratios
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))

        # let's define what are the feature maps that we will
        # use to perform the region of interest cropping, as well as
        # the size of the crop after rescaling.
        # if your backbone returns a Tensor, featmap_names is expected to
        # be [0]. More generally, the backbone should return an
        # OrderedDict[Tensor], and in featmap_names you can choose which
        # feature maps to use.
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=7, #14 if im_size is not None else 7,
                                                        sampling_ratio=2)

        # put the pieces together inside a FasterRCNN model
        model = FasterRCNN(backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)

        if hasattr(model.backbone, 'genotype') and model.backbone.genotype == ViT:
            model.transform = GeneralizedRCNNTransform(int(800 / 2.5), int(1333 / 2.5),
                                                       [0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])

    else:
        # load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        torchvision.models.resnet50()
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    print('Training the object detection model with %d parameters' % (capacity(model)[1]), flush=True)

    return model


if __name__ == "__main__":
    main()
