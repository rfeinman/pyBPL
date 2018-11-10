from __future__ import division, print_function
from abc import ABCMeta, abstractmethod
import warnings
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import torch
import torch.distributions as dist
from abc import ABCMeta, abstractmethod
import os
import sys
import matplotlib.pylab as plt

sys.path.insert(0, os.path.abspath('..'))
from source.other import *
from source.types import *

import imageio
import tqdm

int_types = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]


if __name__ == "__main__":

    lib = Library('../../../lib_data/')
    type_dist = TypeDist(lib)

    def box_only(obj):
        obj.tick_params(
            which='both',
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False
        )

    img_target = imageio.imread('./image_H.jpg')
    img_target = np.asarray(img_target, dtype=np.float32) / 255.

    
    #plt.figure(figsize=(2,2))
    #plt.imshow(img_target, cmap='Greys')
    #box_only(plt)
    #plt.show()


    def get_token_dist():
        # first stroke has 1 sub-stroke, with id "0"
        s1 = Stroke(
            nsub=torch.tensor(1), 
            ids=torch.tensor([0]),
            shapes=lib.shape['mu'][0].view(5, 2, 1),
            invscales=torch.tensor([0.5]),
            lib=lib
        )
        r1 = RelationIndependent(
            category='unihist',
            gpos=torch.tensor([30., -22.]),
            xlim=type_dist.lib.Spatial.xlim,
            ylim=type_dist.lib.Spatial.ylim,
            lib=lib
        )
        # second stroke has 1 sub-stroke, with id "9"
        s2 = Stroke(
            nsub=torch.tensor(1), 
            ids=torch.tensor([9]),
            shapes=lib.shape['mu'][9].view(5, 2, 1),
            invscales=torch.tensor([0.4]),
            lib=lib
        )
        r2 = RelationAttachAlong(
            category='mid',
            attach_ix=0,
            attach_subix=0,
            eval_spot=4.5,
            lib=lib
        )
        # third stroke has 1 sub-stroke, with id "0"
        s3 = Stroke(
            nsub=torch.tensor(1), 
            ids=torch.tensor([0]),
            shapes=lib.shape['mu'][0].view(5, 2, 1),
            invscales=torch.tensor([0.5]),
            lib=lib
        )
        r3 = RelationIndependent(
            category='unihist',
            gpos=torch.tensor([70., -22.]),
            xlim=type_dist.lib.Spatial.xlim,
            ylim=type_dist.lib.Spatial.ylim,
            lib=lib
        )
        k = torch.tensor(3)
        P = [s1, s2, s3]
        R = [r1, r2, r3]

        token_dist = TokenDist(k,P,R,lib)
        
        return token_dist

    token_dist = get_token_dist()
    token = token_dist.sample_token()



    def  get_optimizable_variables(ctype, ctoken, eps):
        assert isinstance(ctype, TokenDist)
        assert isinstance(ctoken, Token_ImDist)
        parameters = []
        lbs = []
        ubs = []
        names = []
        for i in range(ctype.k):
            # shapes type
            ctype.P[i].shapes.requires_grad_()
            parameters.append(ctype.P[i].shapes)
            lbs.append([])
            ubs.append([])
            names.append('shapes_type_%i'%i)
            # shapes token
            ctoken.P[i].shapes.requires_grad_()
            parameters.append(ctoken.P[i].shapes)
            lbs.append([])
            ubs.append([])
            names.append('shapes_token_%i'%i)
            
            # scales type
            ctype.P[i].invscales.requires_grad_()
            parameters.append(ctype.P[i].invscales)
            lbs.append(torch.full(ctype.P[i].invscales.shape, eps))
            ubs.append([])
            names.append('invscales_type_%i'%i)
            # scales token
            ctoken.P[i].invscales.requires_grad_()
            parameters.append(ctoken.P[i].invscales)
            lbs.append(torch.full(ctoken.P[i].invscales.shape, eps))
            ubs.append([])
            names.append('invscales_token_%i'%i)

        return parameters, lbs, ubs, names

    params, lbs, ubs, param_names = get_optimizable_variables(token_dist,token,eps=1e-4)

    token.blur_sigma = 5.

    #plt.figure(figsize=(2,2))
    #plt.imshow(token.pimg.detach().numpy(), cmap='Greys')
    #box_only(plt)
    #plt.show()



    nb_iter = 300
    interval = 30 # how often we will log pimg status
    lr = 5e-4
    img_target = torch.tensor(img_target)

    score_type_list = []
    score_token_list = []
    score_img_list = []
    imgs = []
    param_vals = []
    optimizer = torch.optim.Adam(params, lr=lr)
    for idx in tqdm.tqdm(range(nb_iter)):
        if idx % interval == 0:
            # store pimg at this iteration for later viewing
            imgs.append(np.copy(token.pimg.detach().numpy()))
        # compute scores
        score_type = type_dist.score_type(token_dist)
        score_token = token_dist.score_token(token)
        score_img = token.score_image(img_target)
        score = score_type + score_token + score_img
        # append to lists
        score_type_list.append(score_type)
        score_token_list.append(score_token)
        score_img_list.append(score_img)
        param_vals.append([np.copy(p.detach().numpy()) for p in params])
        # first, zero all gradients
        optimizer.zero_grad()
        # now, perform backward pass
        score_neg = -score
        score_neg.backward()
        # optimization step
        optimizer.step()
        # clip params at boundaries
        with torch.no_grad():
            for ip, param in enumerate(params):
                lb = lbs[ip]
                ub = ubs[ip]
                if len(lb) > 0:
                    torch.max(param, lb, out=param)
                if len(ub) > 0:
                    torch.min(param, ub, out=param)




    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,5))    # type and token scores
    axes[0].plot(score_type_list, c='b', label='log P(type)')
    axes[0].plot(score_token_list, c='g', label='log P(token | type)')
    axes[0].set_ylabel('log-likelihood')
    axes[0].set_xlabel('iteration')
    axes[0].legend()
    # image score
    axes[1].plot(score_img_list, c='r', label='log P(image | token)')
    axes[1].set_ylabel('log-likelihood')
    axes[1].set_xlabel('iteration')
    axes[1].legend()
    plt.show()


    plt.figure(figsize=(2,2))
    plt.imshow(img_target, cmap='Greys')
    box_only(plt)
    plt.title('target')
    plt.show()
    print('')

    fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(15, 2))
    for i in range(10):
        axes[i].imshow(imgs[i], cmap='Greys')
        box_only(axes[i])
        axes[i].set_title('%i' % (interval*i))
    plt.show()


















