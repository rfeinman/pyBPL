"""
Number of strokes model (Kappa)
"""
from __future__ import division, print_function
import numpy as np
import torch
import torch.distributions as dist


def __get_dist(pkappa):
    """
    TODO

    :param pkappa:
    :return:
    """
    assert len(pkappa.shape) == 1
    cat = dist.Categorical(probs=pkappa)

    return cat

def sample_number(libclass, nsamp=1):
    """
    Sample a stroke count, or a vector of stroke counts

    :param libclass: [Library] library class instance
    :param nsamp: [int] number of samples to draw
    :return:
        ns: [(nsamp,) tensor] vector of stroke counts. scalar if nsamp=1.
    """
    # probability of each stroke count
    pkappa = libclass.pkappa
    # get the categorical distribution
    cat = __get_dist(pkappa)
    # sample from the dist
    # NOTE: add 1 to 0-indexed samples
    ns = cat.sample(torch.Size([nsamp])) + 1
    # make sure ns is a vector
    assert len(ns.shape) == 1
    # convert vector to scalar if nsamp=1
    ns = torch.squeeze(ns)

    return ns

def score_number(libclass, ns):
    """
    Score the log-likelihood of each stroke count in ns

    :param libclass: [Library] library class instance
    :param ns: [(nsamp,) tensor] vector of stroke counts. scalar if nsamp=1
    :return:
        ll: [(nsamp,) tensor] vector of log-likelihood scores. scalar if nsamp=1
    """
    # probability of each stroke count
    pkappa = libclass.pkappa
    out_of_bounds = ns > len(pkappa)
    if out_of_bounds.any():
        ll = torch.tensor(-np.inf)
        return ll
    # get the categorical distribution
    cat = __get_dist(pkappa)
    # score points using the categorical distribution
    # NOTE: subtract 1 to get 0-indexed samples
    ll = cat.log_prob(ns-1)

    return ll