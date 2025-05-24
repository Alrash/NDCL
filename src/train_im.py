# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time

import importlib
from stats import SHOT_INDEX

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

import datasets
import hparams_registry
import algorithms
from lib import misc
from lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--type', type=str, default='idg', choices=['normal', 'idg', 'TotalHeavyTail', 'Cross', 'Duality'])
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    if 'DBIDG' == args.algorithm:
        if hparams['batch_size'] > 39:
            hparams['aug_scaler'] = 1
        elif hparams['batch_size'] >= 35:
            hparams['aug_scaler'] = 1.15

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # obtain imbalanced information
    class_show_indicator = None
    if args.type != 'normal':
        module = importlib.import_module(f'stats.{args.dataset}.{args.type}')
        target = ''.join(dataset.ENVIRONMENTS[target] for target in args.test_envs)
        mapping = getattr(module, 'mapping')[target]
        thr = getattr(module, 'thres')[target]
        validation = getattr(module, 'validation')[target]

        class_show_indicator, hparams['env_labels'] = np.zeros(len(dataset[0].classes)), {}
        for k, v in mapping.items():
            hparams['env_labels'][dataset.ENVIRONMENTS.index(k)] = np.zeros_like(class_show_indicator)
            for name, ind in dataset[0].class_to_idx.items():
                class_show_indicator[ind] += v[name]
                hparams['env_labels'][dataset.ENVIRONMENTS.index(k)][ind] = v[name]
        class_show_indicator /= len(dataset) - len(args.test_envs)
        for k in range(class_show_indicator.size):
            if class_show_indicator[k] > thr['many']:
                class_show_indicator[k] = SHOT_INDEX['MANY']
            elif class_show_indicator[k] < thr['few']:
                class_show_indicator[k] = SHOT_INDEX['FEW'] if class_show_indicator[k] != 0 else SHOT_INDEX['ZERO']
            else:
                class_show_indicator[k] = SHOT_INDEX['MEDIAN']
        del module

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    val_splits = []
    for env_i, env in enumerate(dataset):
        uda, val = [], []

        if args.type != 'normal' and env_i not in args.test_envs:
            out, val, in_ = misc.split_dataset_imbalance(env, mapping[dataset.ENVIRONMENTS[env_i]], validation,
                                                         misc.seed_hash(args.trial_seed, env_i))
        else:
            out, in_ = misc.split_dataset(env, int(len(env)*args.holdout_fraction),
                                          misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        val_splits.append((val, None))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    print('Dataset:')
    print('\tENVS:  TRAIN |   TEST |  VALID')
    for i in range(len(dataset)):
        print('\tenv%d: %6d | %6d | %6d' % (i, len(in_splits[i][0]), len(out_splits[i][0]), len(val_splits[i][0])))

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + val_splits + uda_splits)
        if len(env) > 0
    ]

    # loader for online training feature updates
    train_feat_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs
    ] if 'BoDA' in args.algorithm else None


    eval_weights = [None for env, weights in (in_splits + out_splits + val_splits + uda_splits) if len(env) > 0]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_val'.format(i)
        for i in range(len(val_splits)) if len(val_splits[i][0]) != 0]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    hparams['n_steps'] = n_steps - 1

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    if args.type != 'normal':
        del hparams['env_labels']

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))


    last_results_keys = None
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                for x,_ in next(uda_minibatches_iterator)]
        else:
            uda_device = None

        # update features before training step
        if 'BoDA' in args.algorithm and (step > 0 and step % hparams["feat_update_freq"] == 0):
            train_features = {}
            algorithm.eval()
            curr_tr_feats, curr_tr_labels = collections.defaultdict(list), collections.defaultdict(list)
            for name, loader in sorted(zip([f'env{i}' for i in range(len(in_splits))], train_feat_loaders), key=lambda x: x[0]):
                with torch.no_grad():
                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        feats = algorithm.return_feats(x)
                        curr_tr_feats[name].extend(feats.data)
                        curr_tr_labels[name].extend(y.data)
            train_features = {'feats': curr_tr_feats, 'labels': curr_tr_labels}
            algorithm.train()
            step_vals = algorithm.update(minibatches_device, train_features)
        else:
            step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals, class_acc_output = zip(eval_loader_names, eval_loaders, eval_weights), collections.defaultdict(list)
            for name, loader, weights in evals:
                is_test = not (('_in' in name and int(name[3]) not in args.test_envs) or '_val' in name)
                if args.type != 'normal' and is_test:
                    acc, shot_acc, class_acc = misc.accuracy(algorithm, loader, weights, class_show_indicator, device, class_shot_acc=True)
                    results[name+'_acc'] = acc
                    results[name+'_0many'] = np.mean(shot_acc[0])
                    results[name+'_1median'] = np.mean(shot_acc[1])
                    results[name+'_2few'] = np.mean(shot_acc[2])
                    if SHOT_INDEX['ZERO'] in class_show_indicator:
                        results[name+'_3zero'] = np.mean(shot_acc[3])
                    if '_in' in name and int(name[3]) in args.test_envs:
                        class_acc_output[name.split('_')[0]] = list(class_acc)
                else:
                    acc = misc.accuracy(algorithm, loader, weights, class_show_indicator, device, class_shot_acc=False)
                    results[name+'_acc'] = acc

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=15)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=15)

            results.update({
                'hparams': hparams,
                'args': vars(args),
                'class_acc': class_acc_output
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
