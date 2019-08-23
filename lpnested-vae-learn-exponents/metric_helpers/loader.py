# Copyright (c) 2019 Microsoft Corporation.
# Copyright (c) 2018 Ricky Tian Qi Chen
# Licensed under the MIT license.

import torch
import lib.dist as dist
from lib.adapters import LpNestedAdapter
import vae_quant

import ast

def parseISA(isalist):
    print(isalist)
    pnested = []
    pnested.append(isalist[0])
    children = []
    for (p, n) in isalist[1]:
        if n > 0:
            nestedlayer = [p, [ [1.0] ] * n ]
            children.append(nestedlayer)
    if len(children) == 0:
        exit()
    pnested.append(children)
    return pnested

def load_model(checkpt_filename, use_cuda=True):
    print('Loading model and dataset.')
    try:
        checkpt = torch.load(checkpt_filename, map_location=lambda storage, loc: storage)
    except Exception as err:
        print('error: reading file {}'.format(err))
        return None, None
    
    args = checkpt['args']
    state_dict = checkpt['state_dict']

    # backwards compatibility
    if not hasattr(args, 'conv'):
        args.conv = False

    if not hasattr(args, 'pnorm'):
        args.pnorm = 4.0/3.0

    if not hasattr(args, 'q-dist'):
        args.q_dist == 'normal'

    if not hasattr(args, 'var_clipping'):
        args.var_clipping = 0

    # setup the VAE
    if args.dist == 'normal':
        prior_dist = dist.Normal()
    elif args.dist == 'laplace':
        prior_dist = dist.Laplace()
    elif args.dist == 'flow':
        prior_dist = FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
    elif args.dist == 'lpnested':
        if not args.isa == '':
            pnested = parseISA(ast.literal_eval(args.isa))
        elif not args.pnested == '':
            pnested = ast.literal_eval(args.pnested)
        else:
            pnested = parseISA([args.p0, [(args.p1, args.n1), (args.p2, args.n2), (args.p3, args.n3)]])

        print('using Lp-nested prior, pnested = ({}) {}'.format(type(pnested), pnested))
        prior_dist = LpNestedAdapter(p=pnested, scale=args.scale)
        args.latent_dim = prior_dist.dimz()
        print('using Lp-nested prior, changed latent dimension to {}'.format(args.latent_dim))
    elif args.dist == 'studentt':
        print('using student-t prior, scale = {}'.format(args.scale))
        prior_dist = StudentTAdapter(scale=args.scale)
    elif args.dist == 'lpnorm':
        prior_dist = LpNestedAdapter(p=[args.pnorm, [ [1.0] ] * args.latent_dim ], scale=args.scale)

    if args.q_dist == 'normal':
        q_dist = dist.Normal()
    elif args.q_dist == 'laplace':
        q_dist = dist.Laplace()

    vae = vae_quant.VAE(z_dim=args.latent_dim, use_cuda=use_cuda, prior_dist=prior_dist, q_dist=q_dist,
        include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, conv=args.conv, mss=args.mss, var_clipping=(args.var_clipping!=0), dataset=args.dataset)
    vae.load_state_dict(checkpt['state_dict'])
    if use_cuda:
        vae.cuda()
    
    return vae, args
