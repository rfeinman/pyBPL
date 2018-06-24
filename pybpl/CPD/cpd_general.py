"""
Defines the conditional probability distributions that make up the
BPL model
"""
from __future__ import division, print_function
import numpy as np
import torch
import torch.distributions as dist

from pybpl.character.stroke import StrokeType
from .cpd_substrokes import sample_nsub, sample_sequence
from .cpd_shape import sample_shape_type
from .cpd_scale import sample_invscale_type

# ----
# Stroke
# ----

def sample_stroke_type(lib, ns):
    # sample the number of sub-strokes
    nsub = sample_nsub(lib, ns)
    # sample the sub-stroke sequence
    ss_seq = sample_sequence(lib, nsub)
    # sample control points for each sub-stroke in the sequence
    cpts = sample_shape_type(lib, ss_seq)
    # sample scales for each sub-stroke in the sequence
    scales = sample_invscale_type(lib, ss_seq)
    # initialize the stroke type
    stype = StrokeType(ss_seq, cpts, scales)

    return stype

# ----
# Local position model (L)
# ----

def __get_dist(base, sigma_x, sigma_y):
    mu = base
    Cov = torch.eye(2)
    Cov[0,0] = sigma_x
    Cov[1,1] = sigma_y
    mvn = dist.multivariate_normal.MultivariateNormal(mu, Cov)

    return mvn

def sample_position(lib, r, prev_strokes):
    """
    TODO

    :param lib:
    :param r:
    :param prev_strokes:
    :return:
        pos:
    """
    # sample where the position of this stroke should be
    base = r.get_attach_point(prev_strokes)
    assert base.shape == torch.Size([2])
    # get mutlivariate normal distribution
    mvn = __get_dist(base, lib.rel['sigma_x'], lib.rel['sigma_y'])
    # sample position from the distribution
    pos = mvn.sample()

    return pos

def score_position(lib, pos, r, prev_strokes):
    """
    TODO

    :param lib:
    :param pos:
    :param r:
    :param prev_strokes:
    :return:
        ll:
    """
    # sample where the position of this stroke should be
    base = r.get_attach_point(prev_strokes)
    assert base.shape == torch.Size([2])
    # get mutlivariate normal distribution
    mvn = __get_dist(base, lib.rel['sigma_x'], lib.rel['sigma_y'])
    # score position using the distribution
    ll = mvn.log_prob(pos)

    return ll

def sample_affine(lib):
    raise NotImplementedError
    sample_A = Variable(torch.zeros(1, 4))
    # m_scale = Variable(torch.squeeze(torch.Tensor(lib['affine']['mu_scale']))) #broken for some reason
    m_scale = Variable(torch.Tensor([[1, 1]]))
    S_scale = Variable(
        torch.squeeze(torch.Tensor(lib['affine']['Sigma_scale'][0, 0])))

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


    # m_x = Variable(torch.squeeze(torch.Tensor(lib['affine']['mu_xtranslate'])))
    # m_y = Variable(torch.squeeze(torch.Tensor(lib['affine']['mu_ytranslate'])))
    m_x = Variable(torch.zeros(1))
    m_y = Variable(torch.zeros(1))

    s_x = Variable(torch.squeeze(
        torch.Tensor(lib['affine']['sigma_xtranslate'][0, 0])))
    s_y = Variable(torch.squeeze(
        torch.Tensor(lib['affine']['sigma_ytranslate'][0, 0])))

    sample_A[:, 2] = pyro.sample('transx', dist.normal, m_x, s_x)
    sample_A[:, 3] = pyro.sample('transy', dist.normal, m_y, s_y)

    if (sample_A[:, 0:2] <= 0).any():
        print('WARNING: sampled scale variable is less than zero')
    return sample_A


def sample_image(pimg):
    raise NotImplementedError
    I = pyro.sample('image', dist.bernoulli, pimg)  # hope this works.
    return I