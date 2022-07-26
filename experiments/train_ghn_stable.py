# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A wrapper to automatically restart training GHNs from the last saved checkpoint,
e.g. in the case of CUDA OOM or nan loss that can frequently occur due to sampling training architectures.

Example:

    # To train GHN-2 on CIFAR-10:
    python experiments/train_ghn_stable.py experiments/train_ghn.py -m 8 -n -v 50 --ln --name ghn2-cifar10

"""


import os
import sys
from subprocess import PIPE, run

args = sys.argv[1:]
ckpt_args = None
attempts = 0

while attempts < 100:  # let's allow for resuming this job 100 times

    attempts += 1
    print('\nrunning the script time #%d with args:' % attempts, args, ckpt_args, '\n')

    result = run(['python'] + args + ([] if ckpt_args is None else ckpt_args),
                 stderr=PIPE, text=True)

    print('script returned:', result)
    print('\nreturned code:', result.returncode)


    if result.returncode != 0:

        print('Script failed!')

        print('\nERROR:', result.stderr)

        if result.returncode == 2 and result.stderr.find('[Errno 2] No such file or directory') >= 0:
            print('\nRun this script as `python experiments/train_ghn_stable.py experiments/train_ghn.py [args]`\n')
            break

        elif result.stderr.find('RuntimeError') < 0:
            print('\nPlease fix the above printed error and restart the script\n')
            break

        print('restarting the script')
        n1 = result.stderr.find('use this ckpt for resuming the script:')
        if n1 >= 0:
            n1 = result.stderr[n1:].find(':') + n1
            n2 = result.stderr[n1:].find('\n') + n1
            ckpt = result.stderr[n1 + 2 : n2]
            print('parsed path:', ckpt, 'exists:', os.path.exists(ckpt))
            if os.path.exists(ckpt):
                ckpt_args = ['--ckpt', ckpt]
            else:
                print('saved checkpoint file is missing, will be starting from scratch')
        else:
            print('no saved checkpoint found')
        continue
    else:
        break
