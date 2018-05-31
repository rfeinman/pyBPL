"""
CPD for motor programs
"""
from __future__ import division, print_function
import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pybpl.relations import (RelationIndependent, RelationAttach,
                             RelationAttachAlong)
import numpy as np
from pybpl.rendering import bspline_gen_s


# CPD
# functions used in generate_character

def sample_number(libclass):
    pkappa = Variable(torch.squeeze(torch.Tensor(libclass['pkappa'])))

    return pyro.sample('ns', dist.categorical, pkappa, one_hot=False) + 1


def score_number(libclass, ns):  # or multiple
    raise ValueError('not implemented')


def sample_nsub(libclass, ns):
    ptensor = torch.squeeze(torch.Tensor(libclass['pmat_nsub']))
    pvec = Variable(ptensor[ns - 1, :])

    return pyro.sample('nsub', dist.categorical, pvec, one_hot=False) + 1


def sample_relation_type(libclass, prev_strokes):
    nprev = len(prev_strokes)
    stroke_num = nprev + 1
    types = ['unihist', 'start', 'end', 'mid']
    ncpt = 5  # libclass['ncpt'] this may break
    if nprev == 0:
        # ew. this makes it match with indexing
        indx = Variable(torch.LongTensor([0]))
    else:
        mixprob = Variable(
            torch.squeeze(torch.Tensor(libclass['rel']['mixprob'][0, 0]))
        )
        indx = pyro.sample('rtype', dist.categorical, mixprob, one_hot=False)

    rtype = types[indx.data[0]]  # not great practice but whatever

    if rtype == 'unihist':
        # gpos = libclass.Spatial.sample(stroke_num) #TODO
        print('Spatial is unimplemented')
        gpos = Variable(torch.Tensor([4.2, -4.2])).view(-1, 2)
        R = RelationIndependent(rtype, nprev, gpos)

    elif rtype == 'start' or rtype == 'end':
        attach_spot = pyro.sample(
            'attach_spot', dist.categorical, Variable(torch.ones(nprev)),
            one_hot=False
        ) + 1

        R = RelationAttach(rtype, nprev, attach_spot)

    elif rtype == 'mid':
        attach_spot = pyro.sample(
            'attach_spot', dist.categorical, Variable(torch.ones(nprev)),
            one_hot=False
        ) + 1

        nsub = prev_strokes[attach_spot.data[0] - 1].nsub
        subid_spot = pyro.sample(
            'subid_spot', dist.categorical, Variable(torch.ones(nsub)),
            one_hot=False
        ) + 1

        R = RelationAttachAlong(
            rtype, nprev, attach_spot, nsub, subid_spot, ncpt
        )

        # still to be fixed
        _, lb, ub = bspline_gen_s(ncpt, 1)
        R.eval_spot_type = lb + np.random.uniform() * (ub - lb)  # TODO
    else:
        raise TypeError('invalid relation')

    return R


def sample_sequence(libclass, ns, nsub=[]):  # gives template.S[i].ids
    # for i in range(nsamp):

    if nsub == []:
        nsub = sample_nsub(libclass, ns)

    pStart = torch.exp(torch.squeeze(torch.Tensor(libclass['logStart'])))
    sq = [pyro.sample('start_sid', dist.categorical, Variable(pStart),
                      one_hot=False) + 1]

    print('pTransition unimplemented, using uniform transition prob')
    n = int(libclass['N'][0, 0])
    pT = torch.ones(n)

    for bid in range(1, nsub):
        prev = sq[-1]
        # pT = libclass.pT(prev) #need to implement #TODO

        sq.append(pyro.sample('sids', dist.categorical, Variable(pT),
                              one_hot=False) + 1)

    seq = torch.cat(sq, 0)
    # print seq
    return seq


def sample_shape_type(libclass, subid):
    # might want to prepare for when y = any(isnan(libclass.shape.mu))
    k = subid.shape[0]
    # assert type(k) is int
    ncpt = 5
    bspline_stack = Variable(torch.zeros(5, 2, k))

    indx = 0
    for i in subid.data:
        mu = torch.squeeze(torch.Tensor(libclass['shape']['mu'][0, 0]))
        Cov = torch.squeeze(torch.Tensor(libclass['shape']['Sigma'][0, 0]))[:,
              :, i - 1]

        print(
            'Warning: pyro multivariatenormal unimplemented. '
            'using np.random instead ')
        # repair because multivariate sample doesnt work:
        mu = mu.numpy()
        Cov = Cov.numpy()
        rows = np.random.multivariate_normal(mu[i - 1, :], Cov)
        rows = Variable(torch.Tensor(rows))

        # m = torch.distributions.Normal(mu,Cov)
        # rows = m.sample()
        # rows = pyro.sample('sample_shape', dist.multivariatenormal,
        #  Variable(mu[i,:]), Variable(Cov))
        #  this is broken atm

        # print rows

        bspline_stack[:, :, indx] = torch.t(
            rows.view([2, ncpt]))  # this gets the orientation correct
        # print bspline_stack[:,:,indx]
        indx = indx + 1
    return bspline_stack


def sample_invscales_type(libclass, subid):
    # check that it is a vector
    assert len(subid.shape) == 1

    theta = torch.squeeze(torch.Tensor(libclass['scale']['theta'][0, 0]))[
        subid.data - 1]  # because indexing

    # pyro gamma dist uses rate parameter, which is inverse of scale parameter
    invscales = pyro.sample('invscales', dist.gamma, Variable(theta[:, 0]),
                            Variable(1 / theta[:, 1]))
    return invscales


# functions used in generate_exemplar

def sample_relation_token(libclass, eval_spot_type):
    sigma_attach = torch.squeeze(
        torch.Tensor(libclass['tokenvar']['sigma_attach'][0, 0]))
    print("sigma_attach", sigma_attach)
    eval_spot_token = eval_spot_type + sigma_attach.numpy() * \
                                       pyro.sample('randn_for_rtoken',
                                                   dist.normal,
                                                   Variable(torch.zeros(1)),
                                                   Variable(torch.ones(1)))

    ncpt = 5  # TODO
    _, ub, lb = bspline_gen_s(ncpt, 1);  # need to fix
    while eval_spot_token.data[0] < lb or eval_spot_token.data[0] > ub:
        print("lb:", lb)
        print("ub:", ub)
        print("eval_spot_token.data[0]:", eval_spot_token.data[0])
        eval_spot_token = eval_spot_type + sigma_attach.numpy() * \
                                           pyro.sample('randn_for_rtoken',
                                                       dist.normal,
                                                       Variable(torch.zeros(1)),
                                                       Variable(torch.ones(1)))

    return eval_spot_token


def sample_position(libclass, R,
                    prev_strokes):  # check that this does what I want, slicewise

    base = getAttachPoint(R, prev_strokes)
    # indexing into base is not pretty

    sigma_x = Variable(
        torch.squeeze(torch.Tensor(libclass['rel']['sigma_x'][0, 0])))
    sigma_y = Variable(
        torch.squeeze(torch.Tensor(libclass['rel']['sigma_y'][0, 0])))

    x = pyro.sample('r_x_pos', dist.normal, base[0, 0], sigma_x)
    # print 'x', x
    y = pyro.sample('r_y_pos', dist.normal, base[0, 1], sigma_y)
    # print 'y:', y

    pos = torch.stack((x, y), dim=1)
    # print 'pos', pos

    return pos


def sample_shape_token(libclass, bspline_stack):
    sz = bspline_stack.shape
    sigma_shape = torch.squeeze(
        torch.Tensor(libclass['tokenvar']['sigma_shape'][0, 0]))
    outstack = bspline_stack + Variable(sigma_shape) * \
                               pyro.sample('shape_var', dist.normal,
                                           Variable(torch.zeros(sz)),
                                           Variable(torch.ones(sz)))
    return outstack


def sample_invscale_token(libclass, invscales_type):
    # print 'invscales_type', invscales_type
    sz = invscales_type.shape
    sigma_invscale = torch.squeeze(
        torch.Tensor(libclass['tokenvar']['sigma_invscale'][0, 0]))
    invscales_token = invscales_type + Variable(sigma_invscale) * pyro.sample(
        'scales_var', dist.normal, Variable(torch.zeros(sz)),
        Variable(torch.ones(sz)))

    if (invscales_token <= 0).any():
        invscales_token = invscales_type + Variable(
            sigma_invscale) * pyro.sample('scales_var', dist.normal,
                                          Variable(torch.zeros(sz)),
                                          Variable(torch.ones(sz)))
    return invscales_token


def sample_affine(libclass):
    sample_A = Variable(torch.zeros(1, 4))
    # m_scale = Variable(torch.squeeze(torch.Tensor(libclass['affine']['mu_scale']))) #broken for some reason
    m_scale = Variable(torch.Tensor([[1, 1]]))
    S_scale = Variable(
        torch.squeeze(torch.Tensor(libclass['affine']['Sigma_scale'][0, 0])))

    print(
        'Warning: pyro multivariatenormal unimplemented. '
        'using np.random instead')

    m_s = m_scale.data.numpy()[0]
    S_s = S_scale.data.numpy()

    print('m_s[0]:', m_s)
    sample_A[:, 0:2] = Variable(
        torch.Tensor(np.random.multivariate_normal(m_s, S_s)))

    # the actual pyro sample statement:
    # sample_A[:,0:2] = pyro.sample('affine_scale',dist.multivariatenormal,m_scale,S_scale)


    # m_x = Variable(torch.squeeze(torch.Tensor(libclass['affine']['mu_xtranslate'])))
    # m_y = Variable(torch.squeeze(torch.Tensor(libclass['affine']['mu_ytranslate'])))
    m_x = Variable(torch.zeros(1))
    m_y = Variable(torch.zeros(1))

    s_x = Variable(torch.squeeze(
        torch.Tensor(libclass['affine']['sigma_xtranslate'][0, 0])))
    s_y = Variable(torch.squeeze(
        torch.Tensor(libclass['affine']['sigma_ytranslate'][0, 0])))

    sample_A[:, 2] = pyro.sample('transx', dist.normal, m_x, s_x)
    sample_A[:, 3] = pyro.sample('transy', dist.normal, m_y, s_y)

    if (sample_A[:, 0:2] <= 0).any():
        print('WARNING: sampled scale variable is less than zero')
    return sample_A


def sample_image(pimg):
    I = pyro.sample('image', dist.bernoulli, pimg)  # hope this works.
    return I


def getAttachPoint(R, prev_strokes):
    print("prev_strokes:", prev_strokes)
    if R.rtype == 'unihist':
        pos = R.gpos  # make sure this is okay

    elif R.rtype == 'start':

        subtraj = prev_strokes[R.attach_spot.data[0] - 1].motor[
            0]  # this obviously won't work bc no motor attribute rn
        pos = subtraj[0, :]
    elif R.rtype == 'end':
        subtraj = prev_strokes[R.attach_spot.data[0] - 1].motor[
            -1]  # this obviously won't work bc no motor attribute rn
        pos = subtraj[-1, :]
    elif R.rtype == 'mid':
        bspline = prev_strokes[R.attach_spot.data[0] - 1].motor_spline[:, :,
                  R.subid_spot - 1]
        pos = bspline_eval(R.eval_spot_token, bspline)
    else:
        raise ValueError('invalid relation')
    return pos
