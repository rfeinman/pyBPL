"""
Number of strokes model (Kappa)
"""
from __future__ import division, print_function
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
    # sample from the dist. Add 1 to 0-indexed samples
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
    :param ns: [(n,) tensor] vector of stroke counts. scalar if n=1
    :return:
        ll: [(n,) tensor] vector of log-likelihood scores. scalar if n=1
    """

    raise NotImplementedError('not implemented')