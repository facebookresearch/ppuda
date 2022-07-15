# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Trains/fine-tunes a neural network with SGD. Evaluates the network on the image dataset.

Example:

    # Train for 50 epochs as in Tables 2 and 3 of our paper:
    python experiments/sgd/train_net.py --split search --arch 35133 -d cifar10 --epochs 50

    # Train according to the standard NAS protocol:
    python experiments/sgd/train_net.py --split search --arch 35133 -d cifar10 --epochs 600 --cutout --drop_path_prob 0.2 --auxiliary

    # Fine-tune DARTS initialized with the parameters predicted by GHN-2-ImageNet on 100-shot CIFAR-10:
    python experiments/sgd/train_net.py --arch DARTS --epochs 50 -d cifar10 --n_shots 100 --wd 1e-3 --init_channels 48 --layers 14 --ckpt ./checkpoints/ghn2_imagenet.pt

    # Fine-tune ResNet-50 pretrained on Imagenet on 100-shot CIFAR-10:
    python experiments/sgd/train_net.py --split predefined --arch 0 --epochs 50 -d cifar10 --n_shots 100 --wd 1e-3 --pretrained

"""


import torch
import torchvision
import torch.utils
import os
from ppuda.config import init_config
from ppuda.vision.loader import image_loader
from ppuda.deepnets1m.net import Network
from ppuda.deepnets1m.loader import DeepNets1M
from ppuda.deepnets1m.genotypes import ViT, DARTS
import ppuda.deepnets1m.genotypes as genotypes
from ppuda.utils import capacity, adjust_net, infer, pretrained_model, Trainer, init
from ppuda.ghn.nn import GHN


def main():

    args = init_config(mode='train_net')

    is_imagenet = args.dataset == 'imagenet'
    train_queue, valid_queue, num_classes = image_loader(dataset=args.dataset,
                                                         data_dir=args.data_dir,
                                                         test=True,
                                                         load_train_anyway=True,
                                                         batch_size=args.batch_size,
                                                         test_batch_size=args.test_batch_size,
                                                         num_workers=args.num_workers,
                                                         cutout=args.cutout,
                                                         cutout_length=args.cutout_length,
                                                         seed=args.seed,
                                                         noise=args.noise,
                                                         n_shots=args.n_shots)


    assert args.arch is not None, 'architecture genotype/index must be specified'

    try:
        arch = genotype = eval('genotypes.%s' % args.arch)
        net_args = {'C': args.init_channels,
                    'genotype': genotype,
                    'n_cells': args.layers,
                    'C_mult': int(genotype != ViT) + 1,  # assume either ViT or DARTS-style architecture
                    'preproc': genotype != ViT,
                    'stem_type': 1}  # assume that the ImageNet-style stem is used by default
    except (SyntaxError, AttributeError):
        try:
            arch = args.arch
            if args.arch is not None:
                arch = int(args.arch)
            deepnets = DeepNets1M(split=args.split,
                                  nets_dir=args.data_dir,
                                  large_images=is_imagenet,
                                  arch=arch)
            assert len(deepnets) == 1, 'one architecture must be chosen to train'
            graph = deepnets[0]
            net_args, idx = graph.net_args, graph.net_idx
            if 'norm' in net_args and net_args['norm'] == 'bn':
                net_args['norm'] = 'bn-track'
            arch = net_args['genotype']
        except ValueError:
            arch = args.arch.lower()

    if isinstance(arch, str):
        model = adjust_net(eval('torchvision.models.%s(pretrained=%d,num_classes=%d)' %
                                (arch, args.pretrained, 1000 if args.pretrained else num_classes)),
                           is_imagenet or args.imsize > 32)
    else:
        model = Network(num_classes=num_classes,
                        is_imagenet_input=is_imagenet or args.imsize > 32,
                        auxiliary=args.auxiliary,
                        **net_args)

    if (args.ckpt or (args.pretrained and model.__module__.startswith('torchvision.models'))):
        assert bool(args.ckpt is not None) != args.pretrained, 'ckpt and pretrained are mutually exclusive'
        model.expected_input_sz = args.imsize
        model = pretrained_model(model, args.ckpt, num_classes, args.debug, GHN)

    model = init(model,
                 orth=args.init.lower() == 'orth',
                 beta=args.beta,
                 layer=args.layer,
                 max_sz=64,
                 verbose=args.debug > 1)

    model = model.train().to(args.device)

    print('Training arch={} with {} parameters'.format(args.arch, capacity(model)[1]))

    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise NotImplementedError(args.opt)

    if is_imagenet:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.97)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    trainer = Trainer(optimizer,
                      num_classes,
                      is_imagenet,
                      n_batches=len(train_queue),
                      grad_clip=args.grad_clip,
                      auxiliary=args.auxiliary,
                      auxiliary_weight=args.auxiliary_weight,
                      device=args.device,
                      log_interval=args.log_interval,
                      amp=args.amp)

    for epoch in range(max(1, args.epochs)):  # if args.epochs=0, then just evaluate the model

        if args.epochs > 0:
            print('\nepoch={:03d}/{:03d}, lr={:e}'.format(epoch + 1, args.epochs, scheduler.get_last_lr()[0]))
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

            trainer.reset()
            model.train()
            for images, targets in train_queue:
                trainer.update(model, images, targets)
                trainer.log()

            if args.save:
                checkpoint_path = os.path.join(args.save, 'checkpoint.pt')
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, checkpoint_path)
                print('\nsaved the checkpoint to {}'.format(checkpoint_path))


        infer(model.eval(), valid_queue, verbose=True)

        scheduler.step()

if __name__ == '__main__':
    main()
