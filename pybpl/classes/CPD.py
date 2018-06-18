"""
Defines the conditional probability distributions that make up the
BPL model
"""
from __future__ import division, print_function
import warnings
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gamma import Gamma

from .relations import RelationIndependent, RelationAttach, RelationAttachAlong
from ..rendering import bspline_gen_s


# ----
# Number of strokes model (Kappa)
# ----

def sample_number(libclass, nsamp=1):
    """
    Sample a vector of stroke counts

    :param libclass: [Library] library class instance
    :param nsamp: [int] number of samples to draw
    :return:
        ns: [(nsamp,) tensor] vector of stroke counts
    """
    # probability of each stroke count
    pkappa = libclass.pkappa
    # make sure pkappa is a vector
    assert len(pkappa.shape) == 1
    # sample from the categorical distribution. Add 1 to 0-indexed samples
    ns = Categorical(probs=pkappa).sample(torch.Size([nsamp])) + 1
    # make sure ns is a vector
    assert len(ns.shape) == 1

    return ns

def score_number(libclass, ns):
    """
    Score the log-likelihood of each stroke count in ns

    :param libclass: [Library] library class instance
    :param ns: [(n,) tensor] vector of stroke counts
    :return:
        ll: [(n,) tensor] vector of log-likelihood scores
    """
    # make sure ns is a vector
    assert len(ns.shape) == 1

    raise NotImplementedError('not implemented')


# ----
# Sequence of sub-strokes model
# ----

def sample_nsub(libclass, ns):
    """
    Sample the substroke count for each stroke

    :param libclass: [Library] library class instance
    :param ns: [(n,) tensor] vector of stroke counts
    :return:
        nsub: [(n,) tensor] vector of substroke counts
    """
    # make sure ns is a vector
    assert len(ns.shape) == 1
    # probability of each sub-stroke count, conditioned on the number of strokes
    # NOTE: there will be len(ns) probability vectors
    # NOTE: subtract 1 from stroke counts to get Python index
    pvec = libclass.pmat_nsub[ns-1]
    # make sure pvec is a matrix
    assert len(pvec.shape) == 2
    # sample from the categorical distribution. Add 1 to 0-indexed samples
    nsub = Categorical(probs=pvec).sample() + 1
    # make sure nsub is a vector
    assert len(nsub.shape) == 1

    return nsub

def sample_sequence(libclass, ns, nsub=None, nsamp=1):
    """
    Sample the sequence of substrokes

    :param libclass: [Library] library class instance
    :param ns: [tensor] scalar; stroke count
    :param nsub: [tensor] scalar; substroke count
    :param nsamp: [int] number of samples to draw
    :return:
    """
    assert ns.shape == torch.Size([]), \
        "only implemented for scalar 'ns' parameter at the moment"

    if nsub is None:
        # since sample_nsub takes a vector, convert the scalar to a length-1 vec
        ns_vec = ns.view(-1)
        # sample the nsub vector and then convert to scalar
        nsub = sample_nsub(libclass, ns_vec)[0]
    assert nsub.shape == torch.Size([])
    # set pStart variable
    pStart = torch.exp(libclass.logStart)
    warnings.warn(
        'pTransition unimplemented; using uniform transition prob'
    )
    samps = []
    for _ in range(nsamp):
        sq = [Categorical(probs=pStart).sample() + 1]
        for bid in range(1, nsub):
            prev = sq[-1]
            # pT = libclass.pT(prev) #TODO - need to implement
            pT = torch.ones(libclass.N, requires_grad=True)
            sq.append(Categorical(probs=pT).sample() + 1)
        sq = torch.tensor(sq)
        samps.append(sq.view(1,-1))
    samps = torch.cat(samps)

    return samps

# ----
# Relation model (R)
# ----

def sample_relation_type(libclass, prev_strokes):
    """

    :param libclass: [Library] library class instance
    :param prev_strokes: [list of Strokes] list of previous strokes
    :return:
        R: [Relation] relation
    """
    nprev = len(prev_strokes)
    stroke_num = nprev + 1
    types = ['unihist', 'start', 'end', 'mid']
    ncpt = libclass.ncpt
    if nprev == 0:
        indx = torch.tensor(0, dtype=torch.int64, requires_grad=True)
    else:
        indx = Categorical(probs=libclass.rel['mixprob']).sample()

    rtype = types[indx.item()] # TODO - update; this is not great practice

    if rtype == 'unihist':
        # TODO - update SpatialModel class to use torch
        gpos = libclass.Spatial.sample(np.array([stroke_num]))
        R = RelationIndependent(rtype, nprev, gpos)
    elif rtype in ['start', 'end']:
        # sample random attach spot uniformly
        probs = torch.ones(nprev, requires_grad=True)
        attach_spot = Categorical(probs=probs).sample()
        R = RelationAttach(rtype, nprev, attach_spot)
    elif rtype == 'mid':
        # sample random attach spot uniformly
        probs = torch.ones(nprev, requires_grad=True)
        attach_spot = Categorical(probs=probs).sample()
        # sample random subid spot uniformly
        nsub = prev_strokes[attach_spot.item()].nsub
        probs = torch.ones(nsub, requires_grad=True)
        subid_spot = Categorical(probs=probs).sample()

        R = RelationAttachAlong(
            rtype, nprev, attach_spot, nsub, subid_spot, ncpt
        )

        # still to be fixed
        warnings.warn('Setting relation eval_spot_type to be fixed...')
        _, lb, ub = bspline_gen_s(ncpt, 1)
        R.eval_spot_type = lb + np.random.uniform() * (ub - lb)  # TODO
    else:
        raise TypeError('invalid relation')

    return R


# ----
# Shape model (x)
# ----

def sample_shape_type(libclass, subid):
    """

    :param libclass: [Library] library class instance
    :param subid: [(k,) array] vector of sub-stroke ids
    :return:
    """
    # check that it is a vector
    assert len(subid.shape) == 1
    # record vector length
    k = len(subid)
    if isunif(libclass):
        # TODO - update; handle this case
        #bspline_stack = CPDUnif.sample_shape_type(libclass, subid)
        #return bspline_stack
        warnings.warn(
            'isunif=True but CPDUnif not yet implemented... treating as though '
            'isunif=False for now'
        )

    Cov = libclass.shape['Sigma'][:,:,subid].permute([2,0,1])
    mu = libclass.shape['mu'][subid]
    rows_bspline = MultivariateNormal(mu, Cov).sample()
    ncpt = libclass.ncpt
    bspline_stack = torch.zeros((ncpt,2,k), requires_grad=True)
    for i in range(k):
        bspline_stack[:, :, i] = rows_bspline[i].view(ncpt, 2)

    return bspline_stack


# ----
# Scale model (y)
# ----


def sample_invscale_type(libclass, subid):
    """

    :param libclass: [Library] library class instance
    :param subid: [(k,) array] vector of sub-stroke ids
    :return:
    """
    # check that it is a vector
    assert len(subid.shape) == 1

    if isunif(libclass):
        # TODO - update; handle this case
        #invscales = CPDUnif.sample_shape_type(libclass, subid)
        #return invscales
        warnings.warn(
            'isunif=True but CPDUnif not yet implemented... treating as though '
            'isunif=False for now'
        )
    theta = libclass.scale['theta'][subid]
    concentration = theta[:,0]
    # PyTorch gamma dist uses rate parameter, which is inverse of scale
    rate = 1/theta[:,1]
    invscales = Gamma(concentration, rate).sample()

    return invscales





# functions used in generate_exemplar
# TODO - here downward

def sample_relation_token(libclass, eval_spot_type):
    sigma_attach = torch.squeeze(
        torch.Tensor(libclass['tokenvar']['sigma_attach'][0, 0]))
    print("sigma_attach", sigma_attach)
    eval_spot_token = eval_spot_type + sigma_attach.numpy() * \
                                       pyro.sample('randn_for_rtoken',
                                                   dist.normal,
                                                   Variable(torch.zeros(1)),
                                                   Variable(torch.ones(1)))

    ncpt = libclass.ncpt
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

def isunif(libclass):
    """

    :param libclass:
    :return:
    """

    return torch.isnan(libclass.shape['mu']).any()