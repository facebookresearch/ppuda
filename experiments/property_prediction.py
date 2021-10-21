# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Trains and evaluates regression models to predict different properties of architectures.

Example:

    # To run this experiment on CIFAR-10 using our GHN-2:
    python experiments/property_prediction.py cifar10 ./checkpoints/ghn2_cifar10.pt

"""


import numpy as np
import sys, os
import json
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from scipy.stats import kendalltau
from ppuda.ghn.nn import GHN
from ppuda.deepnets1m.loader import DeepNets1M
from ppuda.deepnets1m.net import Network


def main():
    dataset = sys.argv[1]       # cifar10, imagenet
    ckpt = sys.argv[2]          # GHN checkpoint path
    ghn_device = 'cpu'          # little benefit of cuda in this experiment

    is_imagenet = dataset == 'imagenet'

    with open('./data/results_%s.json' % dataset, 'r') as f:
        results = json.load(f)

    properties = {}
    for prop in ['val_acc', 'val_acc_noise', 'time', 'converge_time']:
        properties[prop] = {}
        for split in ['val', 'test']:
            properties[prop][split] = np.array([r[prop] for r in results[split].values()])
    n_train = len(properties['val_acc']['val'])
    assert n_train == len(properties['val_acc']['test']) == 500, \
        ('val ({}) and test ({}) splits are expected to be 500 each'.format(n_train, len(properties['val_acc']['test'])))

    cache_file = ckpt.replace('.pt', '_embed.npy')  # file with graph embeddings
    if os.path.exists(cache_file):
        print('\nloading graph embeddings from the cache: %s' % cache_file)
        x = np.load(cache_file)
        x_train, x_test, x_search = x[:n_train], x[n_train:n_train*2], x[n_train*2:]
    else:
        ghn = GHN.load(ckpt, debug_level=0, device=ghn_device, verbose=True)
        virtual_edges = 50 if ghn.ve else 1  # default values

        def extract_graph_embeddings(graphs_queue):
            x = []
            for graphs in tqdm(graphs_queue):
                assert len(graphs) == 1, ('only one architecture per batch is supported in the evaluation mode', len(graphs))
                net_args, net_idx = graphs.net_args[0], graphs.net_inds[0]

                model = Network(num_classes=1000 if is_imagenet else 10,
                                is_imagenet_input=is_imagenet,
                                **net_args).eval()
                x.append(ghn(model, graphs.to_device(ghn_device), return_embeddings=True)[1].mean(0, keepdim=True).data.cpu().numpy())
            x = np.concatenate(x)
            return x

        print('\nextracting graph embeddings')

        x_train = extract_graph_embeddings(DeepNets1M.loader(split='val', virtual_edges=virtual_edges, large_images=is_imagenet))
        x_test = extract_graph_embeddings(DeepNets1M.loader(split='test', virtual_edges=virtual_edges, large_images=is_imagenet))
        assert len(x_train) == len(x_test) == n_train, (x_train.shape, x_test.shape, n_train)
        x_search = extract_graph_embeddings(DeepNets1M.loader(split='search', virtual_edges=virtual_edges, large_images=is_imagenet))
        np.save(ckpt.replace('.pt', '_embed.npy'), np.concatenate((x_train, x_test, x_search)))


    grid_search_params = {
        'kernel': ['rbf'],
        'C': [1, 10, 50, 10 ** 2, 2 * 10 ** 2, 5 * 10 ** 2, 10 ** 3],
        'gamma': ['auto', 0.05, 0.1, 0.2, 0.5],
        'epsilon': [0.05, 0.1, 0.2]
    }

    for prop, splits in properties.items():
        y_train, y_test = splits['val'], splits['test']

        seeds = [0, 1, 2, 3, None]
        print('\n{}: running the experiment for {} seeds'.format(prop.upper(), len(seeds)))

        scores = []
        for seed in seeds:
            if seed is not None:
                np.random.seed(seed)
                ind_rand = np.random.permutation(n_train)
            else:
                ind_rand = np.arange(n_train)

            # Find the best hyperparameters of SVR on the training set using cross-validation
            clf = GridSearchCV(SVR(), grid_search_params, cv=5, n_jobs=4)
            clf.fit(x_train[ind_rand], y_train[ind_rand])
            if seed is None:
                print('best params', clf.best_params_)

            model = SVR(**clf.best_params_).fit(x_train, y_train)
            y_pred = model.predict(x_test)
            if prop != 'converge_time':
                y_pred = np.round(y_pred)  # rounding slightly improves results
                # in the paper we also round the ground truth values, so the results here can be slightly different

            score = kendalltau(y_test, y_pred)[0]  # rank correlation between prediction and test
            print('Result for seed={}: {:.3f} ({} test samples)'.format(seed, score, len(y_test)))
            scores.append(score)

        print('\nResults for all seeds: {:.3f} +- {:.3f}'.format(np.mean(scores), np.std(scores)))

        x = np.concatenate((x_train, x_test))
        print('Retrain the regression model on {} examples'.format(len(x)))
        model = SVR(**clf.best_params_).fit(x, np.concatenate((y_train, y_test)))  # using the best params found with seed=None

        # Find the best (in the sense of a given property) architecture in the Search split with 100k architectures
        y_pred = model.predict(x_search)
        best_arch = np.argmax(y_pred)
        print('Architecture with the best {} (prediction={:.3f}) in the SEARCH split is {} ({} test samples)'.format(
            prop.upper(), y_pred[best_arch], best_arch, len(y_pred)))

        # the best (in the sense of a given property) architecture can be trained by running (see vision/train_net.py for more examples):
        # python vision/train_net.py --split search --arch $best_arch

    print('\ndone')

if __name__ == '__main__':
    main()
