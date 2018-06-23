"""
Scale model (y)
"""
from __future__ import division, print_function
from torch.distributions.gamma import Gamma

from .cpd_general import isunif
from .. import CPDUnif


def sample_invscale_type(libclass, subid):
    """
    Sample the scale parameters for each sub-stroke

    :param libclass: [Library] library class instance
    :param subid: [(k,) tensor] vector of sub-stroke ids
    :return:
        invscales: [(k,) tensor] vector of scale values
    """
    # check that it is a vector
    assert len(subid.shape) == 1
    # if uniform, sample using CPDUnif and return
    if isunif(libclass):
        invscales = CPDUnif.sample_invscale_type(libclass, subid)
        return invscales
    # create gamma distribution
    theta = libclass.scale['theta'][subid]
    concentration = theta[:,0]
    # NOTE: PyTorch gamma dist uses rate parameter, which is inverse of scale
    rate = 1/theta[:,1]
    gamma = Gamma(concentration, rate)
    # sample from the gamma distribution
    invscales = gamma.sample()

    return invscales

def score_invscale_type(libclass, invscales, subid):
    """
    Score the log-likelihood of each sub-stroke's scale parameter

    :param libclass:
    :param invscales:
    :param subid:
    :return:
    """
    # make sure these are vectors
    assert len(invscales.shape) == 1
    assert len(subid.shape) == 1
    assert len(invscales) == len(subid)
    # if uniform, score using CPDUnif and return
    if isunif(libclass):
        ll = CPDUnif.score_invscale_type(libclass, invscales, subid)
        return ll
    # create gamma distribution
    theta = libclass.scale['theta'][subid]
    concentration = theta[:,0]
    # NOTE: PyTorch gamma dist uses rate parameter, which is inverse of scale
    rate = 1/theta[:,1]
    gamma = Gamma(concentration, rate)
    # score points using the gamma distribution
    ll = gamma.log_prob(invscales)

    return ll

def sample_invscale_token(libclass, invscales_type):
    raise NotImplementedError
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