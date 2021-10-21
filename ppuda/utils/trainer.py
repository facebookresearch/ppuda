# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper to train models.

"""


import time
from torch.nn.parallel import *
from .utils import *
from .darts_utils import *


class Trainer():
    def __init__(self, optimizer, num_classes, is_imagenet, n_batches,
                 grad_clip=5, auxiliary=False, auxiliary_weight=0.4, device='cuda',
                 log_interval=100):
        self.optimizer = optimizer
        criterion = CrossEntropyLabelSmooth(num_classes, 0.1) if is_imagenet else nn.CrossEntropyLoss()
        self.criterion = criterion.to(device[0] if isinstance(device, (list, tuple)) else device)
        self.n_batches = n_batches
        self.grad_clip = grad_clip
        self.auxiliary = auxiliary
        self.auxiliary_weight = auxiliary_weight
        self.device = device
        self.log_interval = log_interval
        self.reset()


    def reset(self):
        self.start = time.time()
        self.step = 0
        self.metrics = {'loss': AvgrageMeter(), 'top1': AvgrageMeter(), 'top5': AvgrageMeter()}


    def update(self, models, images, targets):

        logits = 0
        loss = 0
        count = 0

        self.optimizer.zero_grad()

        if isinstance(self.device, (list, tuple)):
            # Multigpu training
            assert isinstance(models, (list, tuple)) and isinstance(models[0], (list, tuple)), 'models must be a list of lists'
            image_replicas = [images.to(device, non_blocking=True) for device in self.device]
            targets = targets.to(self.device[0], non_blocking=True)  # loss will be computed on the first device

            models_per_device = len(models[0])      # assume that on the first device the number of models is >= than on other devices
            for ind in range(models_per_device):    # for index withing each device
                model_replicas = [models[device][ind] for device in self.device if ind < len(models[device])]
                outputs = parallel_apply(model_replicas,
                                         image_replicas[:len(model_replicas)],
                                         None,
                                         self.device[:len(model_replicas)])  # forward pass at each device in parallel

                # gather outputs from multiple devices and update the loss on the first device
                for device, out in zip(self.device, outputs):
                    y = (out[0] if isinstance(out, (list, tuple)) else out).to(self.device[0])

                    loss += self.criterion(y, targets)
                    if self.auxiliary:
                        loss += self.auxiliary_weight * self.criterion(out[1].to(self.device[0]), targets)
                    logits += y
                    count += 1

        else:

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if not isinstance(models, (list, tuple)):
                models = [models]

            for model in models:
                out = model(images)
                y = out[0] if isinstance(out, tuple) else out

                loss += self.criterion(y, targets)
                if self.auxiliary:
                    loss += self.auxiliary_weight * self.criterion(out[1], targets)

                logits += y
                count += 1

        loss = loss / count         # mean loss across models
        logits = logits / count     # mean logits across models

        if torch.isnan(loss):
            raise RuntimeError('the loss is {}, unable to proceed'.format(loss))

        loss.backward()

        parameters = []
        for group in self.optimizer.param_groups:
            parameters.extend(group['params'])

        nn.utils.clip_grad_norm_(parameters, self.grad_clip)
        self.optimizer.step()

        # Update training metrics
        prec1, prec5 = accuracy(logits, targets, topk=(1, 5))
        n = len(targets)
        self.metrics['loss'].update(loss.item(), n)
        self.metrics['top1'].update(prec1.item(), n)
        self.metrics['top5'].update(prec5.item(), n)

        self.step += 1

        return loss


    def log(self, step=None):
        step_ = self.step if step is None else step
        if step_ % self.log_interval == 0 or step_ >= self.n_batches - 1:
            speed = (time.time() - self.start) / step_
            metrics = '\t'.join(['{}={:.2f}'.format(metric, value.avg) for metric, value in self.metrics.items()])
            print('batch={:04d}/{:04d} \t {} \t {:.2f} sec/batch ({:.1f} min left) '.
                format(step_, self.n_batches,
                       metrics,
                       speed, speed * (self.n_batches - step_) / 60), flush=True)
