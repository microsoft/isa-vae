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
from PIL import Image
import numpy as np
import random

VAR_THRESHOLD = 1e-2


def write_png_greyscale(x_array, filename):
    data = x_array.detach().numpy()
    np.clip(data, 0.0, 1.0, out=data)
    img = np.asarray(data*255.0, dtype=np.dtype(np.uint8))
    img = np.transpose(img, (2,1,0) )
    f = open(filename, 'wb')
    writer = png.Writer(img.shape[1], img.shape[0], greyscale=True, bitdepth=8)
    writer.write_array(f,img.flatten())
    f.close()

def write_png_rgb(x_array, filename):
    img = np.asarray(x_array.detach().numpy()*255.0, dtype=np.dtype(np.uint8))
    img = np.ascontiguousarray(np.transpose(img, (1,2,0) ))
    Image.fromarray(img, 'RGB').save(filename)

def _init_fn(worker_id):
    np.random.seed(0 + worker_id)

def latent_traversal_shapes(vae, shapes_dataset, save, z_inds=None, use_cuda=True, z0=None, prefix=''):
    dataset_loader = DataLoader(shapes_dataset, batch_size=1000, num_workers=0, shuffle=False, worker_init_fn=np.random.seed(0)) # todo shuffle=False

    N = len(dataset_loader.dataset)  # number of data samples
    K = vae.z_dim                    # number of latent variables
    nparams = vae.q_dist.nparams
    vae.eval()

    # print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)

    n = 0
    for xs in dataset_loader:
        batch_size = xs.size(0)
        if use_cuda:
            xs = Variable(xs.view(batch_size, 1, 64, 64).cuda())
        else:
            xs = Variable(xs.view(batch_size, 1, 64, 64))
        qz_params[n:n + batch_size] = vae.encoder.forward(xs).view(batch_size, vae.z_dim, nparams).data
        n += batch_size

    # shape, scale, rotation, x, y, latent, nparams
    qz_params = qz_params.view(3, 6, 40, 32, 32, K, nparams)
    shapes=['square','ellipse','heart']

    # z_j is inactive if Var_x(E[z_j|x]) < eps.
    qz_means = qz_params[:, :, :, :, :, :, 0]
    var = torch.std(qz_means.contiguous().view(N, K), dim=0).pow(2)
    active_units = torch.arange(0, K)[var > VAR_THRESHOLD].long()
    print('Active units: ' + ','.join(map(str, active_units.tolist())))
    n_active = len(active_units)
    print('Number of active units: {}/{}'.format(n_active, vae.z_dim))

    if z_inds is None:
        z_inds = active_units

    if z0 is None:
        z0 = qz_means[:,3,20,16,16,:]
    else:
        if len(z0.shape)==1:
            z0 = torch.unsqueeze(z0, 0)
    print('z0 {}'.format(z0.shape))

    if use_cuda:
        z0 = Variable(z0.cuda())

    # subplots where subplot[i, j] is gt_i vs. z_j
    mean_scale = qz_means.mean(2).mean(2).mean(2)  # (shape, scale, latent)
    mean_rotation = qz_means.mean(1).mean(2).mean(2)  # (shape, rotation, latent)
    mean_pos = qz_means.mean(1).mean(1)  # (shape, pos_x, pos_y, latent)

    # position
    print('mean_pos.shape')
    print(mean_pos.shape)
    print('mean_pos.mean(1).shape')
    print(mean_pos.mean(1).shape)
    print('mean_pos.mean(1).mean(1).shape')
    print(mean_pos.mean(1).mean(1).shape)
    #z0 = mean_pos.mean(1).mean(1)

    z = z0
    z0_mapped = z0.cpu()
    print('z {}'.format(z.shape))
    print('len {}'.format(len(z)))

    x_recon, x_params = vae.decode(z)
    for s in range(len(z)):
        write_png_greyscale(x_recon.cpu()[s,:,:,:], os.path.join(save, 'mean_'+shapes[s]+'.png'))

    z_min= torch.min(torch.min(torch.min(torch.min(qz_means,1)[0],1)[0],1)[0],1)[0]
    z_max = torch.max(torch.max(torch.max(torch.max(qz_means,1)[0],1)[0],1)[0],1)[0]
    z_diff = z_max - z_min

    k_steps = 4
    for i, j in enumerate(z_inds):
        print('z {}'.format(j))
        for k in range(-k_steps,k_steps):
            z = z0_mapped.clone()
            if k<0:
                print('interpolate - from {} by {} towards {}'.format(z0_mapped[:,j], (k/k_steps), z_min[:,j]))
                z[:,j] = z0_mapped[:,j] + (k/k_steps) * (z0_mapped[:,j] - z_min[:,j])
            else:
                print('interpolate + from {} by {} towards {}'.format(z0_mapped[:,j], (k/k_steps), z_min[:,j]))
                z[:,j] = z0_mapped[:,j] + (k/k_steps) * (z_max[:,j] - z0_mapped[:,j])
            print('step {}'.format(k))
            print(z[:,j])

            if use_cuda:
                z = Variable(z.view(3, 1, K).cuda())
            else:
                z = Variable(z.view(3, 1, K))
            x_recon, x_params = vae.decode(z)
            for s in range(len(shapes)):
                write_png_greyscale(x_recon.cpu()[s,:,:,:], os.path.join(save, prefix+shapes[s]+"_z_"+str(i)+"_pos_"+str(k+k_steps)+".png"))


def latent_traversal_cars3d(vae, cars3d_dataset, save, z_inds=None, use_cuda=True, z0=None, prefix=''):
    dataset_loader = DataLoader(cars3d_dataset, batch_size=1000, num_workers=1, shuffle=False)

    n_samples = 10000
    N = min(n_samples, len(dataset_loader.dataset))  # number of data samples
    K = vae.z_dim                    # number of latent variables
    nparams = vae.q_dist.nparams
    vae.eval()

    print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)

    n = 0
    for xs in dataset_loader:
        batch_size = xs.size(0)
        if n+batch_size > N:
            batch_size = N - n
            xs = xs[:batch_size,...]
        xs = Variable(xs.view(batch_size, 3, 64, 64).cuda(), volatile=True)
        qz_params[n:n + batch_size] = vae.encoder.forward(xs).view(batch_size, vae.z_dim, nparams).data
        n += batch_size
        if n>=N:
            break

    if use_cuda:
        qz_params = Variable(qz_params.view(N, K, nparams).cuda())
    else:
        qz_params = Variable(qz_params.view(N, K, nparams))

    # z_j is inactive if Var_x(E[z_j|x]) < eps
    qz_means = qz_params[:, :, 0]
    var = torch.std(qz_means.contiguous().view(N, K), dim=0).pow(2)
    active_units = torch.arange(0, K)[var > VAR_THRESHOLD].long()
    print('Active units: ' + ','.join(map(str, active_units.tolist())))
    n_active = len(active_units)
    print('Number of active units: {}/{}'.format(n_active, vae.z_dim))

    if z_inds is None:
        z_inds = active_units

    if z0 is None:
        z0 = qz_means.mean(0)
    print('z0 {}'.format(z0.shape))

    if use_cuda:
        z0 = Variable(z0.view(K).cuda())
    else:
        z0 = Variable(z0.view(K))

    z = z0
    # x_recon, x_params = vae.decode(z.view(1, 1, K))
    x_recon, x_params = vae.decode(z.view(1, K))
    print('x_recon {}'.format(x_recon.shape))
    write_png_rgb(x_recon.cpu()[0,:,:,:], os.path.join(save, 'mean.png'))

    z_min= torch.min(qz_means,0)[0]
    z_max = torch.max(qz_means,0)[0]
    z_diff = z_max - z_min

    k_steps = 4
    for i, j in enumerate(z_inds):
        print('z {}'.format(j))
        for k in range(-k_steps,k_steps+1):
            z = z0.clone()
            if k<0:
                print('interpolate - from {} by {} towards {}'.format(z0[j], (k/k_steps), z_min[j]))
                z[j] = z0[j] + (k/k_steps) * (z0[j] - z_min[j])
            else:
                print('interpolate + from {} by {} towards {}'.format(z0[j], (k/k_steps), z_max[j]))
                z[j] = z0[j] + (k/k_steps) * (z_max[j] - z0[j])
            print('step {}'.format(k))
            print(z[j])

            if use_cuda:
                # z = Variable(z.view(1, 1, K).cuda())
                z = Variable(z.view(1, K).cuda())
            else:
                # z = Variable(z.view(1, 1, K))
                z = Variable(z.view(1, K))
            x_recon, x_params = vae.decode(z)
            write_png_rgb(x_recon.cpu()[0,:,:,:], os.path.join(save, prefix+"z_"+str(i)+"_"+str(k+k_steps)+".png"))


def latent_traversal_celeba(vae, celeba_dataset, save, z_inds=None, use_cuda=True, z0=None, prefix=''):
    dataset_loader = DataLoader(celeba_dataset, batch_size=1000, num_workers=1, shuffle=False)

    n_samples = 10000
    N = min(n_samples, len(dataset_loader.dataset))  # number of data samples
    K = vae.z_dim                    # number of latent variables
    nparams = vae.q_dist.nparams
    vae.eval()

    print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)

    n = 0
    for xs in dataset_loader:
        batch_size = xs.size(0)
        if n+batch_size > N:
            batch_size = N - n
            xs = xs[:batch_size,...]
        xs = Variable(xs.view(batch_size, 3, 64, 64).cuda(), volatile=True)
        qz_params[n:n + batch_size] = vae.encoder.forward(xs).view(batch_size, vae.z_dim, nparams).data
        n += batch_size
        if n>=N:
            break

    if use_cuda:
        qz_params = Variable(qz_params.view(N, K, nparams).cuda())
    else:
        qz_params = Variable(qz_params.view(N, K, nparams))

    # z_j is inactive if Var_x(E[z_j|x]) < eps
    qz_means = qz_params[:, :, 0]
    var = torch.std(qz_means.contiguous().view(N, K), dim=0).pow(2)
    active_units = torch.arange(0, K)[var > VAR_THRESHOLD].long()
    print('Active units: ' + ','.join(map(str, active_units.tolist())))
    n_active = len(active_units)
    print('Number of active units: {}/{}'.format(n_active, vae.z_dim))

    if z_inds is None:
        z_inds = active_units

    if z0 is None:
        z0 = qz_means.mean(0)
    print('z0 {}'.format(z0.shape))

    if use_cuda:
        z0 = Variable(z0.view(K).cuda())
    else:
        z0 = Variable(z0.view(K))

    z = z0
    # x_recon, x_params = vae.decode(z.view(1, 1, K))
    x_recon, x_params = vae.decode(z.view(1, K))
    print('x_recon {}'.format(x_recon.shape))
    write_png_rgb(x_recon.cpu()[0,:,:,:], os.path.join(save, 'mean.png'))

    z_min= torch.min(qz_means,0)[0]
    z_max = torch.max(qz_means,0)[0]
    z_diff = z_max - z_min

    k_steps = 4
    for i, j in enumerate(z_inds):
        print('z {}'.format(j))
        for k in range(-k_steps,k_steps+1):
            z = z0.clone()
            if k<0:
                print('interpolate - from {} by {} towards {}'.format(z0[j], (k/k_steps), z_min[j]))
                z[j] = z0[j] + (k/k_steps) * (z0[j] - z_min[j])
            else:
                print('interpolate + from {} by {} towards {}'.format(z0[j], (k/k_steps), z_max[j]))
                z[j] = z0[j] + (k/k_steps) * (z_max[j] - z0[j])
            print('step {}'.format(k))
            print(z[j])

            if use_cuda:
                # z = Variable(z.view(1, 1, K).cuda())
                z = Variable(z.view(1, K).cuda())
            else:
                # z = Variable(z.view(1, 1, K))
                z = Variable(z.view(1, K))
            x_recon, x_params = vae.decode(z)
            write_png_rgb(x_recon.cpu()[0,:,:,:], os.path.join(save, prefix+"z_"+str(i)+"_"+str(k+k_steps)+".png"))


def reconstruct_images(vae, xs, save, prefix='', num_channels=1, dimX=64, dimY=64, use_cuda=True, dataset='', data=None):
    if not prefix=='':
        prefix = prefix + '_'
    # reconstruct images
    print('xs.shape')
    print(xs.shape)
    batch_size = len(xs)
    if prefix=='':
        prefix='test'
    show_input = True
    if use_cuda:
        xs = Variable(xs.view(batch_size, num_channels, dimX, dimY).cuda(), volatile=True)
    else:
        xs = Variable(xs.view(batch_size, num_channels, dimX, dimY), volatile=True)
    x_recon, x_params, zs, z_params = vae.reconstruct_img(xs)
    print(xs.shape)
    print(x_recon.shape)
    for i in range(len(x_recon)):
        if num_channels==1:
            if show_input:
                write_png_greyscale(xs.cpu()[i,:,:,:], os.path.join(save, prefix + str(i)+"_input.png"))
            write_png_greyscale(x_recon.cpu()[i,:,:,:], os.path.join(save, prefix + str(i)+"_recon.png"))
        else:
            if show_input:
                write_png_rgb(xs.cpu()[i,:,:,:], os.path.join(save, prefix + str(i)+"_input.png"))
            write_png_rgb(0.01*x_recon.cpu()[i,:,:,:], os.path.join(save, prefix + str(i)+"_recon.png"))
        print('latent traversals')
        if data is not None:
            if dataset=='celeba':
                latent_traversal_celeba(vae, data, save, use_cuda=use_cuda, z0=zs[i,...], prefix=prefix+'sampletraversal_'+str(i)+'_')
            elif dataset=='shapes':
                latent_traversal_shapes(vae, data, save, use_cuda=use_cuda, z0=None, prefix=prefix+'sampletraversal_'+str(i)+'_')
            elif dataset=='cars3d':
                latent_traversal_cars3d(vae, data, save, use_cuda=use_cuda, z0=None, prefix=prefix+'sampletraversal_'+str(i)+'_')


def process_subdirectories(args, use_cuda=True):
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

            dimX=64
            dimY=64
            if args.dataset=='celeba':
                num_channels=3
            elif args.dataset=='cars3d':
                num_channels=3
            else:
                num_channels=1
                
            reconstruct_images(vae, xs, args.save, prefix=dir, num_channels=num_channels, dimX=dimX, dimY=dimY, use_cuda=use_cuda, dataset=cpargs.dataset, data=loader.dataset)


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
    parser.add_argument('--prefix', type=str, default='img-')
    parser.add_argument('-d', '--dataset', default='shapes', type=str, help='dataset name',
        choices=['shapes', 'faces', 'celeba', 'cars3d', '3dchairs', 'ICA'])
    args = parser.parse_args()

    # create output directory if it does not exist
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
        use_cuda = True
    else:
        use_cuda = False

    if args.checkpt!='':
        vae, cpargs = load_model(args.checkpt, use_cuda=use_cuda)
    elif args.directory!='':
        process_subdirectories(args, use_cuda=use_cuda)
        exit(0)
    else:
        print('error: either a checkpoint file or a directory containing experiments must be specified!')
        exit(1)

    dimX=64
    dimY=64
    if args.dataset=='celeba':
        num_channels=3
    elif args.dataset=='cars3d':
        num_channels=3
    else:
        num_channels=1
    print('reconstructing images')
    # dataset loader
    loader = vae_quant.setup_data_loaders(args.dataset, batch_size=16, use_cuda=use_cuda)
    loader = DataLoader(dataset=loader.dataset, batch_size=16, num_workers=0, shuffle=True, pin_memory=True, worker_init_fn=np.random.seed(0))
    imginput = iter(loader)
    xs = imginput.next()
    reconstruct_images(vae, xs, args.save, args.prefix, num_channels=num_channels, dimX=dimX, dimY=dimY, use_cuda=use_cuda, dataset = args.dataset, data=loader.dataset)

    print('latent traversals')
    if cpargs.dataset=='shapes':
        latent_traversal_shapes(vae, loader.dataset, args.save, use_cuda=use_cuda)
    if cpargs.dataset=='celeba':
        latent_traversal_celeba(vae, loader.dataset, args.save, use_cuda=use_cuda)
    if cpargs.dataset=='cars3d':
        latent_traversal_cars3d(vae, loader.dataset, args.save, use_cuda=use_cuda)

