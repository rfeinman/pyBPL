"""
Shape model (x)
"""
from __future__ import division, print_function
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from .cpd_general import isunif
from .. import CPDUnif


def sample_shape_type(libclass, subid):
    """
    Sample the control points for each sub-stroke

    :param libclass: [Library] library class instance
    :param subid: [(nsub,) tensor] vector of sub-stroke ids
    :return:
        bspline_stack: [(ncpt, 2, nsub) tensor] sampled spline
    """
    # check that it is a vector
    assert len(subid.shape) == 1
    # record vector length
    nsub = len(subid)
    # if uniform, sample using CPDUnif and return
    if isunif(libclass):
        bspline_stack = CPDUnif.sample_shape_type(libclass, subid)
        return bspline_stack
    # record num control points
    ncpt = libclass.ncpt
    # create multivariate normal distribution
    Cov = libclass.shape['Sigma'][:,:,subid].permute([2,0,1])
    mu = libclass.shape['mu'][subid]
    mvn = MultivariateNormal(mu, Cov)
    # sample points from the multivariate normal distribution
    rows_bspline = mvn.sample()
    # convert (nsub, ncpt*2) tensor into (ncpt, 2, nsub) tensor
    bspline_stack = torch.transpose(rows_bspline,0,1).view(ncpt,2,nsub)

    return bspline_stack

def score_shape_type(libclass, bspline_stack, subid):
    """
    Score the log-likelihoods of the control points for each sub-stroke

    :param libclass: [Library] library class instance
    :param bspline_stack: [(ncpt, 2, nsub) tensor] shapes of bsplines
    :param subid: [(nsub,) tensor] vector of sub-stroke ids
    :return:
        ll: [(nsub,) tensor] vector of log-likelihood scores
    """
    # check that it is a vector
    assert len(subid.shape) == 1
    # record vector length
    nsub = len(subid)
    assert bspline_stack.shape[-1] == nsub
    # if uniform, score using CPDUnif and return
    if isunif(libclass):
        ll = CPDUnif.score_shape_type(libclass, bspline_stack, subid)
        return ll
    # record num control points
    ncpt = libclass.ncpt
    # convert (ncpt, 2, nsub) tensor into (nsub, ncpt*2) tensor
    rows_bspline = torch.transpose(bspline_stack.view(ncpt * 2, nsub),0,1)
    # create multivariate normal distribution
    Cov = libclass.shape['Sigma'][:,:,subid].permute([2,0,1])
    mu = libclass.shape['mu'][subid]
    mvn = MultivariateNormal(mu, Cov)
    # score points using the multivariate normal distribution
    ll = mvn.log_prob(rows_bspline)

    return ll

def sample_shape_token(libclass, bspline_stack):
    raise NotImplementedError
    sz = bspline_stack.shape
    sigma_shape = torch.squeeze(
        torch.Tensor(libclass['tokenvar']['sigma_shape'][0, 0]))
    outstack = bspline_stack + Variable(sigma_shape) * \
                               pyro.sample('shape_var', dist.normal,
                                           Variable(torch.zeros(sz)),
                                           Variable(torch.ones(sz)))
    return outstack