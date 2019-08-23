# Copyright (c) 2019 Microsoft Corporation.
# Copyright (c) 2018 Ricky Tian Qi Chen
# Licensed under the MIT license.

import math
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable

import lib.utils as utils
from metric_helpers.mi_metric import compute_metric_shapes, compute_metric_faces, compute_metric_celeba

import numpy as np

def estimate_entropies(qz_samples, qz_params, q_dist, n_samples=10000, weights=None, subspaces=None):
    """Computes the term:
        E_{p(x)} E_{q(z|x)} [-log q(z)]
    and
        E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]
    where q(z) = 1/N sum_n=1^N q(z|x_n).
    Assumes samples are from q(z|x) for *all* x in the dataset.
    Assumes that q(z|x) is factorial ie. q(z|x) = prod_j q(z_j|x).

    Computes numerically stable NLL:
        - log q(z) = log N - logsumexp_n=1^N log q(z|x_n)

    Inputs:
    -------
        qz_samples (K, N) Variable
        qz_params  (N, K, nparams) Variable
        weights (N) Variable
    """

    # Only take a sample subset of the samples
    if weights is None:
        qz_samples = qz_samples.index_select(1, Variable(torch.randperm(qz_samples.size(1))[:n_samples].cuda()))
    else:
        sample_inds = torch.multinomial(weights, n_samples, replacement=True)
        qz_samples = qz_samples.index_select(1, sample_inds)

    K, S = qz_samples.size()
    N, _, nparams = qz_params.size()
    assert(nparams == q_dist.nparams)
    assert(K == qz_params.size(1))

    if weights is None:
        weights = -math.log(N)
    else:
        weights = torch.log(weights.view(N, 1, 1) / weights.sum())

    if subspaces is None:
        entropies = torch.zeros(K).cuda()
    else:
        entropies = torch.zeros(len(subspaces)).cuda()

    pbar = tqdm(total=S)
    k = 0
    while k < S:
        batch_size = min(10, S - k)
        logqz_i = q_dist.log_density(
            qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
            qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)[:, :, k:k + batch_size])
        k += batch_size

        if subspaces is None:
            # computes - log q(z_i) summed over minibatch
            entropies += - utils.logsumexp(logqz_i + weights, dim=0, keepdim=False).data.sum(1)
        else:
            for (j,(a,b)) in enumerate(subspaces):
                # computes - log q(z_j) for subspace j summed over minibatch
                logqz_j = torch.sum((logqz_i + weights)[:,a:b,:], dim=1)
                entropies[j] += - utils.logsumexp(logqz_j, dim=0, keepdim=False).data.sum(0)

        pbar.update(batch_size)
    pbar.close()

    entropies /= S

    return entropies


def mutual_info_metric_shapes(vae, shapes_dataset, eval_subspaces=False):
    dataset_loader = DataLoader(shapes_dataset, batch_size=1000, num_workers=0, shuffle=False)

    N = len(dataset_loader.dataset)  # number of data samples
    K = vae.z_dim                    # number of latent variables
    nparams = vae.q_dist.nparams
    vae.eval()

    print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)

    n = 0
    with torch.no_grad():
        for xs in dataset_loader:
            batch_size = xs.size(0)
            xs = Variable(xs.view(batch_size, 1, 64, 64).cuda())
            qz_params[n:n + batch_size] = vae.encoder.forward(xs).view(batch_size, vae.z_dim, nparams).data
            if vae.var_clipping:
                qz_params.data[n:n + batch_size,..., 1] = torch.clamp(qz_params.data[n:n + batch_size,..., 1], vae.lowerbound_ln_var, vae.upperbound_ln_var)
            n += batch_size

    qz_params = Variable(qz_params.view(3, 6, 40, 32, 32, K, nparams).cuda())
    qz_samples = vae.q_dist.sample(params=qz_params)

    print('Estimating marginal entropies.')
    # marginal entropies
    if eval_subspaces:
        subspaces=vae.prior_dist.get_subspaces()
        K_cond_entropies = len(subspaces)
    else:
        subspaces = None
        K_cond_entropies = K

    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams),
        vae.q_dist,
        subspaces=subspaces)

    marginal_entropies = marginal_entropies.cpu()
    cond_entropies = torch.zeros(4, K_cond_entropies)

    print('Estimating conditional entropies for scale.')
    for i in range(6):
        qz_samples_scale = qz_samples[:, i, :, :, :, :].contiguous()
        qz_params_scale = qz_params[:, i, :, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 6, K).transpose(0, 1),
            qz_params_scale.view(N // 6, K, nparams),
            vae.q_dist,
            subspaces=subspaces)

        cond_entropies[0] += cond_entropies_i.cpu() / 6

    print('Estimating conditional entropies for orientation.')
    for i in range(40):
        qz_samples_scale = qz_samples[:, :, i, :, :, :].contiguous()
        qz_params_scale = qz_params[:, :, i, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 40, K).transpose(0, 1),
            qz_params_scale.view(N // 40, K, nparams),
            vae.q_dist,
            subspaces=subspaces)

        cond_entropies[1] += cond_entropies_i.cpu() / 40

    print('Estimating conditional entropies for pos x.')
    for i in range(32):
        qz_samples_scale = qz_samples[:, :, :, i, :, :].contiguous()
        qz_params_scale = qz_params[:, :, :, i, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 32, K).transpose(0, 1),
            qz_params_scale.view(N // 32, K, nparams),
            vae.q_dist,
            subspaces=subspaces)

        cond_entropies[2] += cond_entropies_i.cpu() / 32

    print('Estimating conditional entropies for pox y.')
    for i in range(32):
        qz_samples_scale = qz_samples[:, :, :, :, i, :].contiguous()
        qz_params_scale = qz_params[:, :, :, :, i, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 32, K).transpose(0, 1),
            qz_params_scale.view(N // 32, K, nparams),
            vae.q_dist,
            subspaces=subspaces)

        cond_entropies[3] += cond_entropies_i.cpu() / 32

    metric = compute_metric_shapes(marginal_entropies, cond_entropies)
    return metric, marginal_entropies, cond_entropies


def mutual_info_metric_ICA(vae, ica_dataset):
    dataset_loader = DataLoader(ica_dataset, batch_size=1000, num_workers=1, shuffle=False)

    N = len(dataset_loader.dataset)  # number of data samples
    K = vae.z_dim                    # number of latent variables
    nparams = vae.q_dist.nparams
    vae.eval()

    # print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)

    n = 0
    with torch.no_grad():
        for xs in dataset_loader:
            batch_size = xs.size(0)
            xs = Variable(xs.cuda())
            qz_params[n:n + batch_size] = vae.encoder.forward(xs).view(batch_size, vae.z_dim, nparams).data
            if vae.var_clipping:
                qz_params.data[n:n + batch_size,..., 1] = torch.clamp(qz_params.data[n:n + batch_size,..., 1], vae.lowerbound_ln_var, vae.upperbound_ln_var)
            n += batch_size

    latent_factors = ica_dataset.getLatentFactors()

    qz_params = qz_params.view(N, K, nparams)
    qz_samples = vae.q_dist.sample(params=qz_params)

    print('Estimating marginal entropies.')
    marginal_entropies = torch.zeros(N, K)
    cond_entropies = torch.zeros(4, K)

    metric = 0
    return metric, marginal_entropies, cond_entropies


def mutual_info_metric_faces(vae, shapes_dataset):
    dataset_loader = DataLoader(shapes_dataset, batch_size=1000, num_workers=1, shuffle=False)

    N = len(dataset_loader.dataset)  # number of data samples
    K = vae.z_dim                    # number of latent variables
    nparams = vae.q_dist.nparams
    vae.eval()

    print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)

    n = 0
    with torch.no_grad():
        for xs in dataset_loader:
            batch_size = xs.size(0)
            xs = Variable(xs.view(batch_size, 1, 64, 64).cuda())
            qz_params[n:n + batch_size] = vae.encoder.forward(xs).view(batch_size, vae.z_dim, nparams).data
            if vae.var_clipping:
                qz_params.data[n:n + batch_size,..., 1] = torch.clamp(qz_params.data[n:n + batch_size,..., 1], vae.lowerbound_ln_var, vae.upperbound_ln_var)
            n += batch_size

    qz_params = Variable(qz_params.view(50, 21, 11, 11, K, nparams).cuda())
    qz_samples = vae.q_dist.sample(params=qz_params)

    print('Estimating marginal entropies.')
    # marginal entropies
    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams),
        vae.q_dist)

    marginal_entropies = marginal_entropies.cpu()
    cond_entropies = torch.zeros(3, K)

    print('Estimating conditional entropies for azimuth.')
    for i in range(21):
        qz_samples_pose_az = qz_samples[:, i, :, :, :].contiguous()
        qz_params_pose_az = qz_params[:, i, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_pose_az.view(N // 21, K).transpose(0, 1),
            qz_params_pose_az.view(N // 21, K, nparams),
            vae.q_dist)

        cond_entropies[0] += cond_entropies_i.cpu() / 21

    print('Estimating conditional entropies for elevation.')
    for i in range(11):
        qz_samples_pose_el = qz_samples[:, :, i, :, :].contiguous()
        qz_params_pose_el = qz_params[:, :, i, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_pose_el.view(N // 11, K).transpose(0, 1),
            qz_params_pose_el.view(N // 11, K, nparams),
            vae.q_dist)

        cond_entropies[1] += cond_entropies_i.cpu() / 11

    print('Estimating conditional entropies for lighting.')
    for i in range(11):
        qz_samples_lighting = qz_samples[:, :, :, i, :].contiguous()
        qz_params_lighting = qz_params[:, :, :, i, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_lighting.view(N // 11, K).transpose(0, 1),
            qz_params_lighting.view(N // 11, K, nparams),
            vae.q_dist)

        cond_entropies[2] += cond_entropies_i.cpu() / 11

    metric = compute_metric_faces(marginal_entropies, cond_entropies)
    return metric, marginal_entropies, cond_entropies


def mutual_info_metric_cars3d(vae, shapes_dataset):
    dataset_loader = DataLoader(shapes_dataset, batch_size=1000, num_workers=1, shuffle=False)

    N = len(dataset_loader.dataset)  # number of data samples
    K = vae.z_dim                    # number of latent variables
    nparams = vae.q_dist.nparams
    vae.eval()

    print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)

    n = 0
    with torch.no_grad():
        for xs in dataset_loader:
            batch_size = xs.size(0)
            xs = Variable(xs.view(batch_size, 3, 64, 64).cuda())
            qz_params[n:n + batch_size] = vae.encoder.forward(xs).view(batch_size, vae.z_dim, nparams).data
            if vae.var_clipping:
                qz_params.data[n:n + batch_size,..., 1] = torch.clamp(qz_params.data[n:n + batch_size,..., 1], vae.lowerbound_ln_var, vae.upperbound_ln_var)
            n += batch_size

    qz_params = Variable(qz_params.view(183, 4, 24, K, nparams).cuda())
    qz_samples = vae.q_dist.sample(params=qz_params)

    print('Estimating marginal entropies.')
    # marginal entropies
    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams),
        vae.q_dist)

    marginal_entropies = marginal_entropies.cpu()
    cond_entropies = torch.zeros(3, K)

    print('Estimating conditional entropies for yaw.')
    for i in range(4):
        qz_samples_pose_az = qz_samples[:, i, :, :].contiguous()
        qz_params_pose_az = qz_params[:, i, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_pose_az.view(N // 4, K).transpose(0, 1),
            qz_params_pose_az.view(N // 4, K, nparams),
            vae.q_dist)

        cond_entropies[0] += cond_entropies_i.cpu() / 4

    print('Estimating conditional entropies for pitch.')
    for i in range(24):
        qz_samples_pose_el = qz_samples[:, :, i, :].contiguous()
        qz_params_pose_el = qz_params[:, :, i, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_pose_el.view(N // 24, K).transpose(0, 1),
            qz_params_pose_el.view(N // 24, K, nparams),
            vae.q_dist)

        cond_entropies[1] += cond_entropies_i.cpu() / 24

    metric = compute_metric_faces(marginal_entropies, cond_entropies)
    return metric, marginal_entropies, cond_entropies


def mutual_info_metric_celeba(vae, celeba_dataset):
    dataset_loader = DataLoader(celeba_dataset, batch_size=1000, num_workers=1, shuffle=False)

    n_samples = 10000
    N = min(n_samples, len(dataset_loader.dataset))  # number of data samples
    K = vae.z_dim                    # number of latent variables
    nparams = vae.q_dist.nparams
    vae.eval()

    print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)

    n = 0
    with torch.no_grad():
        for xs in dataset_loader:
            batch_size = xs.size(0)
            if n+batch_size > N:
                batch_size = N - n
                xs = xs[:batch_size,...]
            xs = Variable(xs.view(batch_size, 3, 64, 64).cuda())
            qz_params[n:n + batch_size] = vae.encoder.forward(xs).view(batch_size, vae.z_dim, nparams).data
            if vae.var_clipping:
                qz_params.data[n:n + batch_size,..., 1] = torch.clamp(qz_params.data[n:n + batch_size,..., 1], vae.lowerbound_ln_var, vae.upperbound_ln_var)
            n += batch_size
            if n>=N:
                break

    print('dataset_loader: {} / {}'.format(n,N))

    qz_params = Variable(qz_params.view(N, K, nparams).cuda())
    qz_samples = vae.q_dist.sample(params=qz_params)

    print('Estimating marginal entropies.')
    # marginal entropies
    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams),
        vae.q_dist)

    print('Loading annotations.')
    celeba_annotations = np.loadtxt('../data_celeba/list_attr_celeba.txt', usecols=range(1,41), dtype=int, skiprows=2)
    annotation_names= [ '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young' ]
    n_annotations = celeba_annotations.shape[1]

    marginal_entropies = marginal_entropies.cpu()
    cond_entropies = torch.zeros(n_annotations, K)

    for k in range(0,n_annotations):
        print('{}/{}: Estimating conditional entropies for {}.'.format(k, n_annotations, annotation_names[k]))
        for i in [-1,1]:
            idx = np.where(celeba_annotations[:N,k]==i)
            qz_samples_subset = qz_samples.view(N, K)[idx, :].contiguous()
            qz_params_subset = qz_params.view(N, K, nparams)[idx, :].contiguous()

            n_subset = qz_samples_subset.shape[1]
            print('n_subset {}'.format(n_subset))
            cond_entropies_i = estimate_entropies(
                qz_samples_subset.view(n_subset, K).transpose(0, 1),
                qz_params_subset.view(n_subset, K, nparams),
                vae.q_dist)

            cond_entropies[k] += cond_entropies_i.cpu() / 2

    metric = compute_metric_celeba(marginal_entropies, cond_entropies)
    return metric, marginal_entropies, cond_entropies


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpt', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save', type=str, default='.')
    args = parser.parse_args()

    if args.gpu != 0:
        torch.cuda.set_device(args.gpu)
    vae, dataset, cpargs = load_model_and_dataset(args.checkpt)
    metric, marginal_entropies, cond_entropies = eval('mutual_info_metric_' + cpargs.dataset)(vae, dataset)
    torch.save({
        'metric': metric,
        'marginal_entropies': marginal_entropies,
        'cond_entropies': cond_entropies,
    }, os.path.join(args.save, 'disentanglement_metric.pth'))
    print('MIG: {:.2f}'.format(metric))
