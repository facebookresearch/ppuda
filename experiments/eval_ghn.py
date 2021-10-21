# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluates GHNs on one of the splits of DeepNets-1M.

Examples:

    # Evaluate on predefined architectures:
    python experiments/eval_ghn.py --ckpt ./checkpoints/ghn2_cifar10.pt --split predefined -d cifar10

    # Evaluate on all test architectures:
    python experiments/eval_ghn.py --ckpt ./checkpoints/ghn2_imagenet.pt --split test -d imagenet

    # Evaluate a single architecture (architecture=0) on the wide test split:
    python experiments/eval_ghn.py --ckpt ./checkpoints/ghn2_imagenet.pt --split wide --arch 0 -d imagenet

"""


import torch
import torchvision
import numpy as np
from ppuda.config import init_config
from ppuda.ghn.nn import GHN
from ppuda.vision.loader import image_loader
from ppuda.deepnets1m.loader import DeepNets1M
from ppuda.deepnets1m.net import Network
from ppuda.utils import infer, adjust_net, capacity, AvgrageMeter


args = init_config(mode='eval')

assert args.ckpt is not None, 'please specify the checkpoint for evaluation'
ghn_device = 'cpu'  # use CPU to save GPU memory (useful to avoid OOM when evaluating large models)
ghn = GHN.load(args.ckpt, debug_level=args.debug, device=ghn_device, verbose=True)
virtual_edges = 50 if ghn.ve else 1  # default values

get_loader = lambda test_batch_size: image_loader(args.dataset,
                                                  args.data_dir,
                                                  test=args.split != 'val',  # model selection/tuning is based on validation images and architectures
                                                  test_batch_size=test_batch_size,
                                                  num_workers=args.num_workers,
                                                  noise=args.noise,
                                                  seed=args.seed)[1:]

images_val, num_classes = get_loader(args.test_batch_size)
assert ghn.num_classes == num_classes, ('the evaluation image dataset and the image dataset that the GHN was trained on do not match')

is_imagenet = args.dataset == 'imagenet'

graphs_queue = DeepNets1M.loader(split=args.split,
                                 nets_dir=args.data_dir,
                                 large_images=is_imagenet,
                                 virtual_edges=virtual_edges,
                                 arch=args.arch)

top1_all, top5_all = AvgrageMeter('se'), AvgrageMeter('se')

for graphs in graphs_queue:

    assert len(graphs) == 1, ('only one architecture per batch is supported in the evaluation mode', len(graphs))
    net_args, net_idx = graphs.net_args[0], graphs.net_inds[0]

    if isinstance(net_args['genotype'], str):
        model = adjust_net(eval('torchvision.models.%s(num_classes=%d)' % (net_args['genotype'], num_classes)), is_imagenet).eval()
    else:
        model = Network(num_classes=num_classes,
                        is_imagenet_input=is_imagenet,
                        **net_args).eval()
    print('\nEvaluation of arch={} with {} parameters'.format(net_idx, capacity(model)[1]))

    ghn(model, graphs.to_device(ghn_device))  # predict all the parameters

    for test_batch_size in np.round(np.geomspace(args.test_batch_size, 2, int(np.log2(args.test_batch_size))), 1):
        try:

            if test_batch_size != images_val.batch_size:
                images_val = get_loader(int(test_batch_size))[0]
                print('setting the eval batch size to %d ' % (images_val.batch_size))
            elif args.dataset == 'imagenet':
                images_val.sampler.generator.manual_seed(args.seed)  # set the generator seed to reproduce results

            top1, top5 = infer(model.to(args.device), images_val, verbose=True)

            top1_all.update(top1, 1)
            top5_all.update(top5, 1)

            break

        except RuntimeError as e:
            if str(e).find('CUDA out of memory') >= 0:
                print('CUDA out of memory, attempting to continue with smaller batch size')
                model.to('cpu')  # attempts to release GPU memory
                torch.cuda.empty_cache()
            else:
                raise

print(u'\ndone for {} nets: (avg\u00B1standard error) top1={:.1f}\u00B1{:.1f}, top5={:.1f}\u00B1{:.1f}'.
    format(top1_all.cnt, top1_all.avg, top1_all.dispersion, top5_all.avg, top5_all.dispersion))
