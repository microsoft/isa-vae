# Copyright (c) 2019 Microsoft Corporation.
# Copyright (c) 2018 Ricky Tian Qi Chen
# Licensed under the MIT license.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
import brewer2mpl
bmap = brewer2mpl.get_map('Set1', 'qualitative', 3)
colors = bmap.mpl_colors

plt.style.use('ggplot')

VAR_THRESHOLD = 1e-2


def plot_vs_gt_shapes(vae, shapes_dataset, save, z_inds=None, eval_subspaces=False):
    dataset_loader = DataLoader(shapes_dataset, batch_size=1000, num_workers=1, shuffle=False)

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
            xs = Variable(xs.view(batch_size, 1, 64, 64).cuda())
            qz_params[n:n + batch_size] = vae.encoder.forward(xs).view(batch_size, vae.z_dim, nparams).data
            n += batch_size

    qz_params = qz_params.view(3, 6, 40, 32, 32, K, nparams)

    # z_j is inactive if Var_x(E[z_j|x]) < eps.
    if not eval_subspaces:
        qz_means = qz_params[:, :, :, :, :, :, 0]
    else:
        subspaces=vae.prior_dist.get_subspaces()
        K = len(subspaces)
        qz_means = torch.Tensor(3, 6, 40, 32, 32, K)
        for j, (a,b) in enumerate(subspaces):
            qz_means[...,j] = torch.sum(qz_params[:, :, :, :, :, a:b, 0], -1)
    var = torch.std(qz_means.contiguous().view(N, K), dim=0).pow(2)
    active_units = torch.arange(0, K)[var > VAR_THRESHOLD].long()
    print('Active units: ' + ','.join(map(str, active_units.tolist())))
    n_active = len(active_units)
    print('Number of active units: {}/{}'.format(n_active, vae.z_dim))

    if z_inds is None:
        z_inds = active_units
    
    if len(z_inds) == 0:
        return

    # subplots where subplot[i, j] is gt_i vs. z_j
    mean_scale = qz_means.mean(2).mean(2).mean(2)  # (shape, scale, latent)
    mean_rotation = qz_means.mean(1).mean(2).mean(2)  # (shape, rotation, latent)
    mean_pos = qz_means.mean(0).mean(0).mean(0)  # (pos_x, pos_y, latent)

    fig = plt.figure(figsize=(3, len(z_inds)))  # default is (8,6)
    gs = gridspec.GridSpec(len(z_inds), 3)
    gs.update(wspace=0, hspace=0)  # set the spacing between axes.

    vmin_pos = torch.min(mean_pos)
    vmax_pos = torch.max(mean_pos)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[i * 3])
        ax.imshow(mean_pos[:, :, j].numpy(), cmap=plt.get_cmap('coolwarm'), vmin=vmin_pos, vmax=vmax_pos)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(r'$z_{' + str(j.item()) + r'}$')
        if i == len(z_inds) - 1:
            ax.set_xlabel(r'pos')

    vmin_scale = torch.min(mean_scale)
    vmax_scale = torch.max(mean_scale)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[1 + i * 3])
        ax.plot(mean_scale[0, :, j].numpy(), color=colors[2])
        ax.plot(mean_scale[1, :, j].numpy(), color=colors[0])
        ax.plot(mean_scale[2, :, j].numpy(), color=colors[1])
        ax.set_ylim([vmin_scale, vmax_scale])
        ax.set_xticks([])
        ax.set_yticks([])
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        if i == len(z_inds) - 1:
            ax.set_xlabel(r'scale')

    vmin_rotation = torch.min(mean_rotation)
    vmax_rotation = torch.max(mean_rotation)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[2 + i * 3])
        ax.plot(mean_rotation[0, :, j].numpy(), color=colors[2])
        ax.plot(mean_rotation[1, :, j].numpy(), color=colors[0])
        ax.plot(mean_rotation[2, :, j].numpy(), color=colors[1])
        ax.set_ylim([vmin_rotation, vmax_rotation])
        ax.set_xticks([])
        ax.set_yticks([])
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        if i == len(z_inds) - 1:
            ax.set_xlabel(r'rotation')

    fig.text(0.5, 0.03, 'Ground Truth', ha='center')
    fig.text(0.01, 0.5, 'Learned Latent Variables ', va='center', rotation='vertical')
    plt.savefig(save)
    plt.close()


def plot_vs_gt_ICA(vae, ica_dataset, save, z_inds=None):
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
            n += batch_size

    latent_factors = ica_dataset.getLatentFactors()

    qz_params = qz_params.view(N, K, nparams)

    # z_j is inactive if Var_x(E[z_j|x]) < eps.
    qz_means = qz_params[:, :, 0]
    var = torch.std(qz_means.contiguous().view(N, K), dim=0).pow(2)
    active_units = torch.arange(0, K)[var > VAR_THRESHOLD].long()
    print('Active units: ' + ','.join(map(str, active_units.tolist())))
    n_active = len(active_units)
    print('Number of active units: {}/{}'.format(n_active, vae.z_dim))

    if z_inds is None:
        z_inds = active_units

    if len(z_inds) == 0:
        return

    # subplots where subplot[i, j] is gt_i vs. z_j
    mean_scale = qz_means

    fig = plt.figure(figsize=(6, 2*len(z_inds)))  # default is (8,6)
    gs = gridspec.GridSpec(len(z_inds), 3)
    gs.update(wspace=0, hspace=0)  # set the spacing between axes.

    vmin_scale = torch.min(mean_scale)
    vmax_scale = torch.max(mean_scale)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[i * 3])
        ax.scatter(latent_factors[:,0].numpy(), mean_scale[:, j].numpy(), color=colors[2], s=0.5)
        ax.set_ylim([vmin_scale, vmax_scale])
        ax.set_xticks([])
        ax.set_yticks([])
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        ax.set_ylabel(r'$z_{' + str(j.item()) + r'}$')
        if i == len(z_inds) - 1:
            ax.set_xlabel(r'F1')

    vmin_scale = torch.min(mean_scale)
    vmax_scale = torch.max(mean_scale)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[1 + i * 3])
        ax.scatter(latent_factors[:,1].numpy(), mean_scale[:, j].numpy(), color=colors[0], s=0.5)
        ax.set_ylim([vmin_scale, vmax_scale])
        ax.set_xticks([])
        ax.set_yticks([])
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        if i == len(z_inds) - 1:
            ax.set_xlabel(r'F2')

    fig.text(0.5, 0.03, 'Ground Truth', ha='center')
    fig.text(0.01, 0.5, 'Learned Latent Variables ', va='center', rotation='vertical')

    # for plotting the true and learned weights during fitting as well as the objective
    
    W = ica_dataset.getW()
    x = ica_dataset.getX()
    W_est = vae.decoder.W.weight.detach().cpu().numpy()

    ax2 = fig.add_subplot(gs[2])

    ax2.scatter(0.1*x[:,0],0.1*x[:,1],s=0.5, color='g')

    for i, j in enumerate(z_inds):
        ax2.plot([W_est[0,j],0,-W_est[0,j]], [W_est[1,j],0,-W_est[1,j]], color = 'red',linewidth=1.0, label='original weight')

    ax2.plot([W[0,0],0,-W[0,0]], [W[1,0],0,-W[1,0]], color = 'blue',linewidth=1.0, label='original weight')
    ax2.plot([W[0,1],0,-W[0,1]], [W[1,1],0,-W[1,1]], color = 'blue',linewidth=1.0, label='original weight')
    ax2.plot([W[0,2],0,-W[0,2]], [W[1,2],0,-W[1,2]], color = 'blue',linewidth=1.0, label='original weight')

    ax2.axis('equal')

    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
                      
    plt.tight_layout()

    plt.savefig(save)
    plt.close()

    # save W and W_est and the cosine between them
    filename_w = os.path.splitext(save)[0]+'_W.pth'
    torch.save({
        'W': W,
        'W_est': W_est,
        'z_inds': z_inds
    }, filename_w)



def plot_vs_gt_faces(vae, faces_dataset, save, z_inds=None):
    dataset_loader = DataLoader(faces_dataset, batch_size=1000, num_workers=1, shuffle=False)

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
            xs = Variable(xs.view(batch_size, 1, 64, 64).cuda())
            qz_params[n:n + batch_size] = vae.encoder.forward(xs).view(batch_size, vae.z_dim, nparams).data
            n += batch_size

    qz_params = qz_params.view(50, 21, 11, 11, K, nparams)

    # z_j is inactive if Var_x(E[z_j|x]) < eps.
    qz_means = qz_params[:, :, :, :, :, 0]
    var = torch.std(qz_means.contiguous().view(N, K), dim=0).pow(2)
    active_units = torch.arange(0, K)[var > VAR_THRESHOLD].long()
    print('Active units: ' + ','.join(map(str, active_units.tolist())))
    n_active = len(active_units)
    print('Number of active units: {}/{}'.format(n_active, vae.z_dim))

    if z_inds is None:
        z_inds = active_units

    if len(z_inds) == 0:
        return

    # subplots where subplot[i, j] is gt_i vs. z_j
    mean_pose_az = qz_means.mean(3).mean(2).mean(0)  # (pose_az, latent)
    mean_pose_el = qz_means.mean(3).mean(1).mean(0)  # (pose_el, latent)
    mean_light_az = qz_means.mean(2).mean(1).mean(0)  # (light_az, latent)

    fig = plt.figure(figsize=(len(z_inds), 3))  # default is (8,6)
    gs = gridspec.GridSpec(3, len(z_inds))
    gs.update(wspace=0, hspace=0)  # set the spacing between axes.

    vmin_scale = torch.min(mean_pose_az)
    vmax_scale = torch.max(mean_pose_az)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[i])
        ax.plot(mean_pose_az[:, j].numpy())
        ax.set_ylim([vmin_scale, vmax_scale])
        ax.set_xticks([])
        ax.set_yticks([])
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        if i == 0:
            ax.set_ylabel(r'azimuth')

    vmin_scale = torch.min(mean_pose_el)
    vmax_scale = torch.max(mean_pose_el)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[len(z_inds) + i])
        ax.plot(mean_pose_el[:, j].numpy())
        ax.set_ylim([vmin_scale, vmax_scale])
        ax.set_xticks([])
        ax.set_yticks([])
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        if i == 0:
            ax.set_ylabel(r'elevation')

    vmin_scale = torch.min(mean_light_az)
    vmax_scale = torch.max(mean_light_az)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[2 * len(z_inds) + i])
        ax.plot(mean_light_az[:, j].numpy())
        ax.set_ylim([vmin_scale, vmax_scale])
        ax.set_xticks([])
        ax.set_yticks([])
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        if i == 0:
            ax.set_ylabel(r'lighting')
        ax.set_xlabel(r'z{}'.format(j))

    plt.suptitle('GT Factors vs. Latent Variables')
    plt.savefig(save)
    plt.close()


def plot_vs_gt_cars3d(vae, cars3d_dataset, save, z_inds=None):
    dataset_loader = DataLoader(cars3d_dataset, batch_size=1000, num_workers=1, shuffle=False)

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
            xs = Variable(xs.view(batch_size, 3, 64, 64).cuda())
            qz_params[n:n + batch_size] = vae.encoder.forward(xs).view(batch_size, vae.z_dim, nparams).data
            n += batch_size

    qz_params = qz_params.view(183, 4, 24, K, nparams)

    # z_j is inactive if Var_x(E[z_j|x]) < eps.
    qz_means = qz_params[:, :, :, :, 0]
    var = torch.std(qz_means.contiguous().view(N, K), dim=0).pow(2)
    active_units = torch.arange(0, K)[var > VAR_THRESHOLD].long()
    print('Active units: ' + ','.join(map(str, active_units.tolist())))
    n_active = len(active_units)
    print('Number of active units: {}/{}'.format(n_active, vae.z_dim))

    if z_inds is None:
        z_inds = active_units

    if len(z_inds) == 0:
        return

    # subplots where subplot[i, j] is gt_i vs. z_j
    mean_pose_az = qz_means.mean(2).mean(0)  # (pitch, latent)
    mean_pose_el = qz_means.mean(1).mean(0)  # (yaw, latent)

    fig = plt.figure(figsize=(len(z_inds), 3))  # default is (8,6)
    gs = gridspec.GridSpec(3, len(z_inds))
    gs.update(wspace=0, hspace=0)  # set the spacing between axes.

    vmin_scale = torch.min(mean_pose_az)
    vmax_scale = torch.max(mean_pose_az)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[i])
        ax.plot(mean_pose_az[:, j].numpy())
        ax.set_ylim([vmin_scale, vmax_scale])
        ax.set_xticks([])
        ax.set_yticks([])
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        if i == 0:
            ax.set_ylabel(r'pitch')

    vmin_scale = torch.min(mean_pose_el)
    vmax_scale = torch.max(mean_pose_el)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[len(z_inds) + i])
        ax.plot(mean_pose_el[:, j].numpy())
        ax.set_ylim([vmin_scale, vmax_scale])
        ax.set_xticks([])
        ax.set_yticks([])
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        if i == 0:
            ax.set_ylabel(r'yaw')
        ax.set_xlabel(r'z{}'.format(j))

    plt.suptitle('GT Factors vs. Latent Variables')
    plt.savefig(save)
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpt', required=True)
    parser.add_argument('-zs', type=str, default=None)
    parser.add_argument('-gpu', type=int, default=0)
    parser.add_argument('-save', type=str, default='latent_vs_gt.pdf')
    parser.add_argument('-elbo_decomp', action='store_true')
    args = parser.parse_args()

    from elbo_decomposition import elbo_decomposition
    import lib.dist as dist
    import lib.flows as flows
    from vae_quant import VAE, setup_data_loaders

    def load_model_and_dataset(checkpt_filename):
        print('Loading model and dataset.')
        checkpt = torch.load(checkpt_filename, map_location=lambda storage, loc: storage)
        args = checkpt['args']
        state_dict = checkpt['state_dict']

        # model
        if not hasattr(args, 'dist') or args.dist == 'normal':
            prior_dist = dist.Normal()
            q_dist = dist.Normal()
        elif args.dist == 'laplace':
            prior_dist = dist.Laplace()
            q_dist = dist.Laplace()
        elif args.dist == 'flow':
            prior_dist = flows.FactorialNormalizingFlow(dim=args.latent_dim, nsteps=4)
            q_dist = dist.Normal()
        vae = VAE(z_dim=args.latent_dim, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist, conv=args.conv)
        vae.load_state_dict(state_dict, strict=False)

        # dataset loader
        loader = setup_data_loaders(args)
        return vae, loader, args

    z_inds = list(map(int, args.zs.split(','))) if args.zs is not None else None
    torch.cuda.set_device(args.gpu)
    vae, dataset_loader, cpargs = load_model_and_dataset(args.checkpt)
    if args.elbo_decomp:
        elbo_decomposition(vae, dataset_loader)
    eval('plot_vs_gt_' + cpargs.dataset)(vae, dataset_loader.dataset, args.save, z_inds)
