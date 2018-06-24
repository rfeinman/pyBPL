"""
Defines the conditional probability distributions that make up the
BPL model
"""
from __future__ import division, print_function
import numpy as np
import torch

from ..stroke import StrokeType
from ..relations import get_attach_point
from .cpd_substrokes import sample_nsub, sample_sequence
from .cpd_shape import sample_shape_type
from .cpd_scale import sample_invscale_type


def sample_stroke_type(libclass, ns):
    # sample the number of sub-strokes
    nsub = sample_nsub(libclass, ns)
    # sample the sub-stroke sequence
    ss_seq = sample_sequence(libclass, nsub)
    # sample control points for each sub-stroke in the sequence
    cpts = sample_shape_type(libclass, ss_seq)
    # sample scales for each sub-stroke in the sequence
    scales = sample_invscale_type(libclass, ss_seq)
    # initialize the stroke type
    stype = StrokeType(ss_seq, cpts, scales)

    return stype

def sample_position(libclass, R,
                    prev_strokes):  # check that this does what I want, slicewise
    raise NotImplementedError
    base = get_attach_point(R, prev_strokes)
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

def sample_affine(libclass):
    raise NotImplementedError
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
    raise NotImplementedError
    I = pyro.sample('image', dist.bernoulli, pimg)  # hope this works.
    return I