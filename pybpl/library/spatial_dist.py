"""
Spatial distribution class definition.
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist


class SpatialDist(object):
    """
    A distribution for spatial locations of relations.
    """
    def __init__(
            self, xlim, ylim
    ):
        """
        Build a 2D histogram model of the data

        :param xlim: [list of 2 ints] (xmin,xmax); range of x-dimension
        :param ylim: [list of 2 ints] (ymin,ymax); range of y-dimension
        """
        assert len(xlim) == 2
        assert len(ylim) == 2

        self.xlim = xlim
        self.ylim = ylim
        self.dist = None

    def fit(self, data):
        assert len(data.shape) == 2
        ndata, dim = data.shape
        assert dim == 2
        # TODO
        self.dist = 1

    def set_properties(self):
        """
        Set the properties of the SpatialDist instance manually

        """
        low = torch.tensor([self.xlim[0], self.ylim[0]], dtype=torch.float)
        high = torch.tensor([self.xlim[1], self.ylim[1]], dtype=torch.float)
        self.dist = dist.uniform.Uniform(low=low, high=high)

    def sample(self, nsamp):
        """
        Sample from a 2D histogram model

        :param nsamp: number of samples
        :return:
            samples: [(n,2) array] samples
            yi: [(n,) array] y-bin index
            xi: [(n,) array] x-bin index
        """
        if self.dist is None:
            raise Exception('SpatialDist must be fit before sampling.')
        samples = self.dist.sample(torch.Size([nsamp]))

        return samples

    def score(self, data):
        """
        Compute the log-likelihood of data

        :param data: [(n,2) tensor] data to model
        :return:
            ll: [(n,) tensor] log-likelihood scores
        """
        if self.dist is None:
            raise Exception('SpatialDist must be fit before scoring.')
        # check the input
        assert len(data.shape) == 2
        n, dim = data.shape
        assert dim == 2
        # compute log-probabilities
        ll = self.dist.log_prob(data)
        # TODO - update this when we are no longer using Uniform
        ll = torch.sum(ll, dim=1)

        return ll

    def plot(self, subplot=False):
        """
        Visualize the learned position model

        :param subplot: [bool] whether this is a subplot of a larger figure
        :return: None
        """
        N, _, _ = np.histogram2d(data[:, 0], data[:, 1], bins=edges)
        pYX = torch.exp(self.logpYX)
        img = pYX / torch.max(pYX)
        if subplot:
            plt.imshow(img.numpy(), cmap='gray', origin='lower')
        else:
            plt.figure()
            plt.imshow(img.numpy(), cmap='gray', origin='lower')
            plt.show()