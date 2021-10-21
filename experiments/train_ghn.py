# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Trains Graph HyperNetwork on DeepNets-1M.

Example:

    # To train GHN-2 on CIFAR-10:
    python experiments/train_ghn.py -m 8 -n -v 50 --ln --name ghn2-cifar10

"""


import torch
import os
from torch.optim.lr_scheduler import MultiStepLR
from ppuda.config import init_config
from ppuda.ghn.nn import GHN, ghn_parallel
from ppuda.vision.loader import image_loader
from ppuda.deepnets1m.loader import DeepNets1M
from ppuda.utils import capacity, Trainer
from ppuda.deepnets1m.net import Network


def main():

    args = init_config(mode='train_ghn')

    train_queue, val_queue, num_classes = image_loader(args.dataset,
                                                       args.data_dir,
                                                       test=False,
                                                       batch_size=args.batch_size,
                                                       test_batch_size=args.test_batch_size,
                                                       num_workers=args.num_workers,
                                                       seed=args.seed)

    is_imagenet = args.dataset == 'imagenet'
    graphs_queue = DeepNets1M.loader(args.meta_batch_size,
                                     split=args.split,
                                     nets_dir=args.data_dir,
                                     virtual_edges=args.virtual_edges,
                                     num_nets=args.num_nets,
                                     large_images=is_imagenet)

    ghn = GHN(max_shape=args.max_shape,
              num_classes=num_classes,
              hypernet=args.hypernet,
              decoder=args.decoder,
              weight_norm=args.weight_norm,
              ve=args.virtual_edges > 1,
              layernorm=args.ln,
              hid=args.hid,
              debug_level=args.debug).to(args.device)
    if args.multigpu:
        ghn = ghn_parallel(ghn)


    optimizer = torch.optim.Adam(ghn.parameters(), args.lr, weight_decay=args.wd)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.gamma)

    trainer = Trainer(optimizer,
                      num_classes,
                      is_imagenet,
                      n_batches=len(train_queue),
                      grad_clip=args.grad_clip,
                      device=ghn.device_ids if args.multigpu else args.device,
                      log_interval=args.log_interval)


    seen_nets = set()

    print('\nStarting training GHN with {} parameters!'.format(capacity(ghn)[1]))

    for epoch in range(args.epochs):

        print('\nepoch={:03d}/{:03d}, lr={:e}'.format(epoch + 1, args.epochs, scheduler.get_last_lr()[0]))

        trainer.reset()
        ghn.train()
        failed_batches = 0

        for step, (images, targets) in enumerate(train_queue):

            upd, loss = False, torch.zeros(1, device=args.device)
            while not upd:
                try:
                    graphs = next(graphs_queue)

                    nets_torch = []

                    for nets_args in graphs.net_args:
                        net = Network(is_imagenet_input=is_imagenet,
                                      num_classes=num_classes,
                                      compress_params=True,
                                      **nets_args)
                        nets_torch.append(net)

                    # Predict parameters
                    nets_torch = ghn(nets_torch, graphs if args.multigpu else graphs.to_device(args.device))
                    loss = trainer.update(nets_torch, images, targets)
                    trainer.log()

                    for ind in graphs.net_inds:
                        seen_nets.add(ind)
                    upd = True

                except RuntimeError as e:
                    print('error', type(e), e)
                    oom = str(e).find('out of memory') >= 0
                    is_nan = torch.isnan(loss) or str(e).find('the loss is') >= 0
                    if oom or is_nan:
                        if failed_batches > len(train_queue) // 50:
                            print('Out of patience (after %d attempts to continue), '
                                  'please restart the job with another seed !!!' % failed_batches)
                            raise

                        if oom:
                            print('CUDA out of memory, attempt to clean memory #%d' % failed_batches)
                            if args.multigpu:
                                ghn = ghn.module
                            ghn.to('cpu')
                            torch.cuda.empty_cache()
                            ghn.to(args.device)
                            if args.multigpu:
                                ghn = ghn_parallel(ghn)

                        failed_batches += 1

                    else:
                        raise

            del images, targets, graphs, nets_torch, loss
            if step % 10 == 0:
                torch.cuda.empty_cache()

        if args.save:
            # Save config necessary to restore GHN configuration when evaluating it
            config = {}
            config['max_shape'] = args.max_shape
            config['num_classes'] = num_classes
            config['hypernet'] = args.hypernet
            config['decoder'] = args.decoder
            config['weight_norm'] = args.weight_norm
            config['ve'] = (ghn.module if args.multigpu else ghn).ve
            config['layernorm'] = args.ln
            config['hid'] = args.hid

            checkpoint_path = os.path.join(args.save, 'ghn.pt')
            torch.save({'state_dict': (ghn.module if args.multigpu else ghn).state_dict(),
                        'optimzer': optimizer.state_dict(),
                        'epoch': epoch,
                        'config': config}, checkpoint_path)
            print('\nsaved the checkpoint to {}'.format(checkpoint_path))

        print('{} unique architectures seen'.format(len(seen_nets)))

        scheduler.step()

        # Evaluation is done in a separate script: eval_ghn.py


if __name__ == '__main__':
    main()
