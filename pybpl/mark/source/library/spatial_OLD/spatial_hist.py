"""
Spatial histogram class definition.
"""
from __future__ import print_function, division
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions.uniform import Uniform
from torch.distributions.categorical import Categorical

from ...util import aeq, ind2sub, logsumexp_t

class SpatialHist(object):
    """
    A 2D plane is divided into an evenly spaced grid, where a square is chosen
    randomly and then points are chosen uniformly from the square.
    """
    def __init__(
            self, data=None, xlim=None, ylim=None,
            nbin_per_side=None, prior_count=None
    ):
        """
        Build a 2D histogram model of the data

        :param data: [(n,2) tensor] data to model
        :param xlim: [list of 2 ints] (xmin,xmax); range of x-dimension
        :param ylim: [list of 2 ints] (ymin,ymax); range of y-dimension
        :param nbin_per_side: [int] number of bins per dimension
        :param prior_count: [float] prior counts in each cell (not added to
                            edge cells)
        """
        # if params are empty, return; model properties will be set later
        # using 'set_properties' method
        params = [data, xlim, ylim, nbin_per_side]
        if all([item is None for item in params]):
            return

        # set default value of 0 for prior_count
        if prior_count is None:
            prior_count = 0.

        ndata, dim = data.shape
        assert len(xlim) == 2
        assert len(ylim) == 2
        assert dim == 2

        # compute the "edges" of the histogram
        xtick = torch.linspace(xlim[0], xlim[1], nbin_per_side+1)
        ytick = torch.linspace(ylim[0], ylim[1], nbin_per_side+1)
        assert len(xtick)-1 == nbin_per_side
        assert len(ytick)-1 == nbin_per_side
        edges = [xtick, ytick]

        # length, in pixels, of a side of a bin
        rg_bin = torch.tensor(
            [(xlim[1] - xlim[0]), (ylim[1] - ylim[0])],
            dtype=torch.float32
        )
        rg_bin = rg_bin / nbin_per_side

        # Compute the histogram
        N = myhist3(data, edges)
        diff = ndata - torch.sum(N)
        if diff > 0:
            warnings.warn('%i position points are out of bounds' % diff)

        # Add in the prior counts
        N = torch.transpose(N, 0, 1)
        N = N + prior_count
        logN = torch.log(N)

        # Convert to probability distribution
        logpN = logN - logsumexp_t(logN)
        assert aeq(torch.sum(torch.exp(logpN)), torch.tensor(1.))

        self.logpYX = logpN
        self.xlab = xtick
        self.xlim = xtick[[0,-1]]
        self.ylab = ytick
        self.ylim = ytick[[0,-1]]
        self.rg_bin = rg_bin
        self.prior_count = prior_count

    def set_properties(self, logpYX, xlab, ylab, rg_bin, prior_count):
        """
        Set the properties of the SpatialHist instance manually

        :param logpYX: [(m,n) tensor] TODO
        :param xlab: [(m,) tensor] TODO
        :param ylab: [(n,) tensor] TODO
        :param rg_bin: [(2,) tensor] TODO
        :param prior_count: [float] TODO
        :return: None
        """
        for elt in [logpYX, xlab, ylab, rg_bin]:
            assert isinstance(elt, torch.Tensor)
        assert type(prior_count) is float
        self.logpYX = logpYX
        self.xlab = xlab
        self.xlim = xlab[[0,-1]]
        self.ylab = ylab
        self.ylim = ylab[[0,-1]]
        self.rg_bin = rg_bin
        self.prior_count = prior_count

    def sample(self, nsamp):
        """
        Sample from a 2D histogram model

        :param nsamp: number of samples
        :return:
            samples: [(n,2) array] samples
            yi: [(n,) array] y-bin index
            xi: [(n,) array] x-bin index
        """
        assert type(nsamp) is int
        # Pick which bins the samples are from
        # NOTE: transpose here so that torch.Tensor.view(-1) is equivalent
        # to matlab's A(:) functionality
        logpvec = torch.transpose(self.logpYX, 0, 1)
        logpvec = logpvec.contiguous().view(-1)
        pvec = torch.exp(logpvec)
        pvec = pvec / torch.sum(pvec)
        lin = Categorical(probs=pvec).sample(torch.Size([nsamp]))

        # Retrieve the [x, y] indices of these bins
        yi, xi = ind2sub(self.logpYX.shape, lin)

        # Retrieve the edges for each of these bins
        xmin = self.xlab[xi]
        ymin = self.ylab[yi]
        xmax = self.xlab[xi+1]
        ymax = self.ylab[yi+1]
        assert len(xmin) == len(xmax)
        assert len(ymin) == len(ymax)

        # Sample from a uniform distribution in each of the bins
        xsamp = Uniform(low=xmin, high=xmax).sample(torch.Size([1]))
        ysamp = Uniform(low=ymin, high=ymax).sample(torch.Size([1]))
        samples = torch.transpose(torch.cat([xsamp, ysamp], 0), 0, 1)

        return samples, yi, xi

    def score(self, data):
        """
        Compute the log-likelihood of data under a 2D histogram model

        :param data: [(n,2) tensor] data to model
        :return:
            ll: [(n,) tensor] log-likelihood scores
        """
        # Compute bin in histogram
        n, dim = data.shape
        edges = [self.xlab, self.ylab]

        mylogpYX = self.logpYX

        # fast classification
        ll = fast_hclassif(data, mylogpYX, edges)

        # Adjust log-likelihoods to account for the uniform component
        # of the data
        ll = ll - n*torch.log(self.rg_bin[0]) - n*torch.log(self.rg_bin[1])
        assert not torch.isnan(ll).any()

        return ll

    def get_id(self, data):
        """
        TODO - description

        :param data: [(n,2) tensor] data to model
        :return:
            id: [(n,2) tensor] x and y id of each point in bins
            ll: [(n,) tensor] log-likelihood of each point
        """
        n, dim = data.shape
        edges = [self.xlab, self.ylab]
        ll = torch.zeros(n)
        xid = torch.zeros(n)
        yid = torch.zeros(n)
        mylogpYX = self.logpYX
        for i in range(n):
            ll[i], xid[i], yid[i] = hclassif(data[i:i+1], mylogpYX, edges)
        id = torch.cat([xid.view(-1,1), yid.view(-1,1)], 1)
        ll = ll - torch.log(self.rg_bin[0]) - torch.log(self.rg_bin[1])
        assert not np.any(np.isnan(ll))

        return id, ll

    def plot(self, subplot=False):
        """
        Visualize the learned position model

        :param subplot: [bool] whether this is a subplot of a larger figure
        :return: None
        """
        pYX = torch.exp(self.logpYX)
        img = pYX / torch.max(pYX)
        if subplot:
            plt.imshow(img.numpy(), cmap='gray', origin='lower')
        else:
            plt.figure()
            plt.imshow(img.numpy(), cmap='gray', origin='lower')
            plt.show()


def hclassif(pt, logpYX, edges):
    """
    Compute the log-likelihood of the point "pt"

    :param pt:
    :param logpYX:
    :param edges:
    :return:
        logprob:
        xid:
        yid:
    """
    N = myhist3(pt, edges)
    N = torch.transpose(N, 0, 1)
    N = N > 0
    ind = torch.nonzero(N)
    xid = ind[:,0]
    yid = ind[:,1]
    if torch.sum(N) == 0:
        logprob = torch.tensor(-np.inf)
        xid = 1
        yid = 1
    else:
        logprob = logpYX[N]
        assert not torch.isnan(logprob)

    return logprob, xid, yid


def fast_hclassif(pt, logpYX, edges):
    """
    Vectorized version of hclassif

    :param pt:
    :param logpYX:
    :param edges:
    :return:
        logprob:
    """
    npt = len(pt)
    N = myhist3(pt, edges)
    N = torch.transpose(N, 0, 1)
    sumN = torch.sum(N)
    MTPL = N * logpYX
    MTPL[N==0] = 0
    logprob = torch.sum(MTPL)
    if sumN < npt:
        logprob = torch.tensor(-np.inf)
    assert not torch.isnan(logprob)

    return logprob


def myhist3(data, edges):
    """
    Since PyTorch doesn't have a histogram function, we'll have to do histogram
    in numpy and then convert back

    :param data: [(n,2) tensor] data to model
    :param edges: [list of 2 tensors] (array array); the x and y bins
    :return:
        N: [(m,n) tensor] modified histogram
    """
    # TODO:     update this function to remain fully in torch, rather than
    # TODO:     converting back & forth to numpy
    # convert to numpy
    data = data.numpy()
    edges = [edges[0].numpy(), edges[1].numpy()]
    # Cluster with histogram function
    N, _, _ = np.histogram2d(data[:,0], data[:,1], bins=edges)

    # convert back to torch
    N = torch.tensor(N, dtype=torch.float32)

    return N