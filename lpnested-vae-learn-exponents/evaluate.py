# Copyright (c) 2019 Microsoft Corporation.
# Copyright (c) 2018 Ricky Tian Qi Chen
# Licensed under the MIT license.

import math
import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import lib.utils as utils
from metric_helpers.loader import load_model
import vae_quant

import png
import numpy as np
import random

from elbo_decomposition import elbo_decomposition
from plot_latent_vs_true import plot_vs_gt_shapes, plot_vs_gt_faces, plot_vs_gt_cars3d, plot_vs_gt_ICA  # noqa: F401

import disentanglement_metrics

def evaluate(args, outputdir, vae, dataset, prefix=''):
    if os.path.exists(os.path.join(outputdir, prefix+'combined_data.pth')):
        return

    # Report statistics
    vae.eval()
    dataset_loader = DataLoader(dataset, batch_size=1000, num_workers=0, shuffle=True)

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    logpx, dependence, information, dimwise_kl, analytical_cond_kl, elbo_marginal_entropies, elbo_joint_entropy = \
        elbo_decomposition(vae, dataset_loader)
    torch.save({
        'args': args,
        'logpx': logpx,
        'dependence': dependence,
        'information': information,
        'dimwise_kl': dimwise_kl,
        'analytical_cond_kl': analytical_cond_kl,
        'marginal_entropies': elbo_marginal_entropies,
        'joint_entropy': elbo_joint_entropy
    }, os.path.join(outputdir, prefix+'elbo_decomposition.pth'))

    metric, metric_marginal_entropies, metric_cond_entropies = eval('disentanglement_metrics.mutual_info_metric_' + args.dataset)(vae, dataset_loader.dataset, eval_subspaces=False)
    torch.save({
        'args': args,
        'metric': metric,
        'marginal_entropies': metric_marginal_entropies,
        'cond_entropies': metric_cond_entropies,
    }, os.path.join(outputdir, prefix+'disentanglement_metric.pth'))
    print('logpx: {:.2f}'.format(logpx))
    print('MIG: {:.2f}'.format(metric))

    eval('plot_vs_gt_' + args.dataset)(vae, dataset_loader.dataset, os.path.join(outputdir, prefix+'gt_vs_latent.png'))

    torch.save({
        'args': args,
        'logpx': logpx,
        'dependence': dependence,
        'information': information,
        'dimwise_kl': dimwise_kl,
        'analytical_cond_kl': analytical_cond_kl,
        'elbo_marginal_entropies': elbo_marginal_entropies,
        'elbo_joint_entropy': elbo_joint_entropy,
        'metric': metric,
        'metric_marginal_entropies': metric_marginal_entropies,
        'metric_cond_entropies': metric_cond_entropies,
    }, os.path.join(outputdir, prefix+'combined_data.pth'))

    if args.dist == 'lpnested':
        eval('plot_vs_gt_' + args.dataset)(vae, dataset_loader.dataset, os.path.join(outputdir, 'gt_vs_grouped_latent.png'), eval_subspaces=True)

        metric_subspaces, metric_marginal_entropies_subspaces, metric_cond_entropies_subspaces = eval('disentanglement_metrics.mutual_info_metric_' + args.dataset)(vae, dataset_loader.dataset, eval_subspaces=True)
        torch.save({
            'args': args,
            'metric': metric_subspaces,
            'marginal_entropies': metric_marginal_entropies_subspaces,
            'cond_entropies': metric_cond_entropies_subspaces,
        }, os.path.join(outputdir, 'disentanglement_metric_subspaces.pth'))
        print('MIG grouped by subspaces: {:.2f}'.format(metric_subspaces))

        torch.save({
            'args': args,
            'logpx': logpx,
            'dependence': dependence,
            'information': information,
            'dimwise_kl': dimwise_kl,
            'analytical_cond_kl': analytical_cond_kl,
            'elbo_marginal_entropies': elbo_marginal_entropies,
            'elbo_joint_entropy': elbo_joint_entropy,
            'metric': metric,
            'metric_marginal_entropies': metric_marginal_entropies,
            'metric_cond_entropies': metric_cond_entropies,
            'metric_subspaces': metric_subspaces,
            'metric_marginal_entropies_subspaces': metric_marginal_entropies_subspaces,
            'metric_cond_entropies_subspaces': metric_cond_entropies_subspaces
        }, os.path.join(outputdir, 'combined_data.pth'))



def process_subdirectories(args, outputdir, use_cuda=True):
    loader = vae_quant.setup_data_loaders(args.dataset, batch_size=16, use_cuda=use_cuda)
    loader = DataLoader(dataset=loader.dataset, batch_size=16, num_workers=0, shuffle=True, pin_memory=True, worker_init_fn=np.random.seed(0))
    imginput = iter(loader)
    xs = imginput.next()
    for dirpath, dirnames, filenames in os.walk(args.directory):
        for i,dir in enumerate(dirnames):
            checkpt = os.path.join(args.directory,dir,'checkpt-0000.pth')
            print(checkpt)
            vae, cpargs = load_model(checkpt, use_cuda=use_cuda)
            if vae is None:
                continue
            if cpargs.dataset!=args.dataset:
                print("error: model was trained on different dataset: {}".format(cpargs.dataset))
                exit(1)
            if loader==None:
                loader = vae_quant.setup_data_loaders(cpargs.dataset, batch_size=16, use_cuda=use_cuda)

            this_outputdir = os.path.join(outputdir, dir)
            evaluate(cpargs, this_outputdir, vae, loader.dataset, prefix='')


# python vis_traversal.py --directory ../paramsearch-celeba-10-eu2-mseloss/ --gpu -1 --dataset celeba --save ../test-celeba-images-mse2/

if __name__ == '__main__':
    # for deterministic results
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic=True

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpt', type=str, default='')
    parser.add_argument('--directory', type=str, default='')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save', type=str, default='.')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('-d', '--dataset', default='shapes', type=str, help='dataset name',
        choices=['shapes', 'faces', 'celeba', 'cars3d', '3dchairs', 'ICA'])
    args = parser.parse_args()

    # create output directory if it does not exist
    outputdir = args.save
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
        use_cuda = True
    else:
        use_cuda = False

    if args.checkpt!='':
        vae, cpargs = load_model(args.checkpt, use_cuda=use_cuda)
    elif args.directory!='':
        process_subdirectories(args, outputdir, use_cuda=use_cuda)
        exit(0)
    else:
        print('error: either a checkpoint file or a directory containing experiments must be specified!')
        exit(1)

    # dataset loader
    loader = vae_quant.setup_data_loaders(args.dataset, batch_size=16, use_cuda=use_cuda)

    evaluate(cpargs, outputdir, vae, loader.dataset)
