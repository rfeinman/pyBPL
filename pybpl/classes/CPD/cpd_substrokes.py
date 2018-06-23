"""
Sequence of sub-strokes model (z)
"""
from __future__ import division, print_function
import torch
from torch.distributions.categorical import Categorical


def sample_nsub(libclass, ns, nsamp=1):
    """
    Sample a sub-stroke count (or a vector of sub-stroke counts if nsamp>1)

    :param libclass: [Library] library class instance
    :param ns: [tensor] stroke count. scalar
    :param nsamp: [int] number of samples to draw
    :return:
        nsub: [(n,) tensor] vector of sub-stroke counts. scalar if nsamp=1
    """
    # probability of each sub-stroke count, conditioned on the number of strokes
    # NOTE: subtract 1 from stroke counts to get Python index
    pvec = libclass.pmat_nsub[ns-1]
    # make sure pvec is a vector
    assert len(pvec.shape) == 1
    # sample from the categorical distribution. Add 1 to 0-indexed samples
    nsub = Categorical(probs=pvec).sample(torch.Size([nsamp])) + 1
    # convert vector to scalar if nsamp=1
    nsub = torch.squeeze(nsub)

    return nsub

def sample_sequence(libclass, nsub, nsamp=1):
    """
    Sample the sequence of sub-strokes for this stroke

    :param libclass: [Library] library class instance
    :param nsub: [tensor] scalar; sub-stroke count
    :param nsamp: [int] number of samples to draw
    :return:
        samps: [(nsamp, nsub) tensor] matrix of sequence samples. vector if
                nsamp=1
    """
    # nsub should be a scalar
    assert nsub.shape == torch.Size([])

    samps = []
    for _ in range(nsamp):
        # set initial transition probabilities
        pT = torch.exp(libclass.logStart)
        # sub-stroke sequence is a list
        seq = []
        # step through and sample 'nsub' sub-strokes
        for _ in range(nsub):
            # sample the sub-stroke
            ss = Categorical(probs=pT).sample()
            seq.append(ss)
            # update transition probabilities; condition on previous sub-stroke
            pT = libclass.pT(ss)
        # convert list into tensor
        seq = torch.tensor(seq)
        samps.append(seq.view(1,-1))
    # concatenate list of samples into tensor (matrix)
    samps = torch.cat(samps)
    # if nsamp=1 this should be a vector
    samps = torch.squeeze(samps, dim=0)

    return samps