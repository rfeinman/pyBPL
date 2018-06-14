"""
Spatial histogram class definition.
"""
from __future__ import print_function, division
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp


class SpatialHist(object):
    """
    A 2D plane is divided into an evenly spaced grid, where a square is chosen
    randomly and then points are chosen uniformly from the square.
    """
    def __init__(self, data, xlim, ylim, nbin_per_slide, prior_count=0):
        """
        Build a 2D histogram model of the data

        :param data: [(n,2) array] data to model
        :param xlim: [list of 2 ints] (xmin,xmax); range of x-dimension
        :param ylim: [list of 2 ints] (ymin,ymax); range of y-dimension
        :param nbin_per_slide: [int] number of bins per dimension
        :param prior_count: [float] prior counts in each cell (not added to
                            edge cells)
        """
        ndata, dim = data.shape
        assert len(xlim) == 2
        assert len(ylim) == 2
        assert dim == 2

        # compute the "edges" of the histogram
        xtick = np.linspace(xlim[0], xlim[1], nbin_per_slide+1)
        ytick = np.linspace(ylim[0], ylim[1], nbin_per_slide+1)
        assert len(xtick)-1 == nbin_per_slide
        assert len(ytick)-1 == nbin_per_slide
        edges = [xtick, ytick]

        # Store important information about the bins
        self.rg_bin = [
            (xlim[1] - xlim[0]) / nbin_per_slide,
            (ylim[1] - ylim[0]) / nbin_per_slide
        ] # length, in pixels, of a side of a bin
        self.xlab = xtick
        self.ylab = ytick

        # Compute the histogram
        N = myhist3(data, edges)
        diff = ndata - np.sum(N)
        if diff > 0:
            warnings.warn('%i position points are out of bounds' % diff)

        # Add in the prior counts
        N = np.transpose(N)
        N = N + prior_count
        logN = np.log(N)

        # Convert to probability distribution
        logpN = logN - logsumexp(logN)
        #assert aeq(np.sum(np.exp(logpN)),1) # TODO - what is "aeq"?

        self.logpYX = logpN
        self.xlab = xtick
        self.ylab = ytick
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
        # Pick which bins the samples are from
        logpvec = self.logpYX.flatten()
        pvec = np.exp(logpvec)
        pvec = pvec / np.sum(pvec)
        lin = np.zeros(nsamp, dtype=int)
        for i in range(nsamp):
            x = np.random.multinomial(1, pvec)
            lin[i] = np.nonzero(x)[0][0]

        # Retrieve the [y, x] indices of these bins
        yi, xi = np.unravel_index(lin, self.logpYX.shape)

        # Retrieve the edges for each of these bins
        xmin = self.xlab[xi]
        ymin = self.ylab[yi]
        xmax = self.xlab[xi+1]
        ymax = self.ylab[yi+1]

        # Sample from a uniform distribution in each of the bins
        xsamp = np.multiply(xmax-xmin, np.random.uniform(size=nsamp)) + xmin
        ysamp = np.multiply(ymax-ymin, np.random.uniform(size=nsamp)) + ymin
        samples = np.transpose(np.vstack([xsamp, ysamp]))

        return samples, yi, xi

    def score(self, data):
        """
        Compute the log-likelihood of data under a 2D histogram model

        :param data: [(n,2) array] data to model
        :return:
            ll: [(n,) array] log-likelihood scores
        """
        # Compute bin in histogram
        n, dim = data.shape
        edges = [self.xlab, self.ylab]

        mylogpYX = self.logpYX

        # fast classification
        ll = fast_hclassif(data, mylogpYX, edges)

        # Adjust log-likelihoods to account for the uniform component
        # of the data
        ll = ll - n*np.log(self.rg_bin[0]) - n*np.log(self.rg_bin[1])
        assert not np.any(np.isnan(ll))

        return ll

    def get_id(self, data):
        """
        TODO - description

        :param data: [(n,2) array] data to model
        :return:
            id: [(n,2) array] x and y id of each point in bins
            ll: [(n,) array] log-likelihood of each point
        """
        n, dim = data.shape
        edges = [self.xlab, self.ylab]
        ll = np.zeros(n)
        xid = np.zeros(n)
        yid = np.zeros(n)
        mylogpYX = self.logpYX
        for i in range(n):
            ll[i], xid[i], yid[i] = hclassif(data[i:i+1], mylogpYX, edges)
        id = np.transpose(np.vstack([xid, yid]))
        ll = ll - np.log(self.rg_bin[0]) - np.log(self.rg_bin[1])
        assert not np.any(np.isnan(ll))

        return id, ll

    def plot(self, subplot=False):
        """
        Visualize the learned position model

        :param subplot: [bool] whether this is a subplot of a larger figure
        :return: None
        """
        pYX = np.exp(self.logpYX)
        img = pYX / np.max(pYX)
        if subplot:
            plt.imshow(img, cmap='gray')
        else:
            plt.figure(figsize=(10,8))
            plt.imshow(img, cmap='gray')
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
    N = np.transpose(N)
    N = N.astype(np.bool)
    yid, xid = np.where(N)
    if np.sum(N) == 0:
        logprob = -np.inf
        xid = 1
        yid = 1
    else:
        logprob = logpYX[N]
        assert not np.isnan(logprob)

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
    N = np.transpose(N)
    sumN = np.sum(N)
    MTPL = np.multiply(N, logpYX)
    MTPL[N==0] = 0
    logprob = np.sum(MTPL)
    if sumN < npt:
        logprob = -np.inf
    assert not np.isnan(logprob)

    return logprob


def myhist3(data, edges):
    """
    Modified histogram function, where datapoints on the edge are mapped to
    the last cell, not their own cell

    :param data: [(n,2) array] data to model
    :param edges: [list of 2 arrays] (array array); the x and y bins
    :return:
        N:
    """
    # Cluster with histogram function
    N, _, _ = np.histogram2d(data[:,0], data[:,1], bins=edges)
    N = N.astype(np.int32)

    # Move the last row/col to the second to last
    lastcol = N[:,-1]
    lastrow = N[-1,:]
    last = N[-1,-1]
    N[:,-2] = N[:,-2] + lastcol
    N[-2,:] = N[-2,:] + lastrow
    #N[-2,-2] = N[-2,-2] + last # <--- TODO: is this necessary?

    # Delete last row and column
    N = np.delete(N, -1, axis=0)
    N = np.delete(N, -1, axis=1)

    return N
