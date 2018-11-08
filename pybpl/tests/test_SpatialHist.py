import unittest
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform

from ..library import SpatialHist
from ..util import logsumexp_t

class TestSpatialHist(unittest.TestCase):

    def setUp(self):
        # Parameters
        nbin_per_side = 50
        prior_count = 0.1
        xlm = [-30, 20]
        ylm = [-15, 15]

        n = 1000  # number of training points in each slice

        # Shape of distribution
        mu1 = torch.tensor([-22., 0.])
        mu2 = torch.tensor([0., 8.])
        Sigma = torch.eye(2)

        # Sample the data
        data1 = MultivariateNormal(mu1, Sigma).sample(torch.Size([n]))
        data2 = MultivariateNormal(mu2, Sigma).sample(torch.Size([n]))
        data = torch.cat([data1, data2])

        # Build the SpatialHist instance & sample data
        self.H = SpatialHist(data, xlm, ylm, nbin_per_side, prior_count)
        self.syndata, _, _ = self.H.sample(n)

        self.data = data
        self.xlim = xlm
        self.ylim = ylm

    def test_sample(self):
        """
        Plot the sampled data next to the original data to verify that it
        looks correct.
        """
        # Plot original data
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,8))
        axes[0].scatter(self.data[:,0], self.data[:,1], s=3, c='r')
        axes[0].set_title('original data')
        axes[0].set_xlim(self.xlim)
        axes[0].set_ylim(self.ylim)
        # plot reconstructed data
        axes[1].scatter(self.syndata[:,0], self.syndata[:,1], s=3, c='b')
        axes[1].set_title('reconstructed data')
        axes[1].set_xlim(self.xlim)
        axes[1].set_ylim(self.ylim)
        plt.show()
        # TODO - write assertions

    def test_plot(self):
        """
        Test visualization of learned position model
        """
        self.H.plot()
        # TODO - write assertions

    def test_dualMethodLL(self):
        """
        Check two different ways of computing likelihood
        """
        ll = self.H.score(self.syndata)
        _, ll2 = self.H.get_id(self.syndata)
        ll2 = torch.sum(ll2)
        self.assertTrue(torch.abs(ll - ll2) <= 1e-2)

    def test_validDensity(self):
        """
        Numerically check the normalizing constant of the density
        """
        nsamp = 10000
        area = (self.xlim[1]-self.xlim[0]) * (self.ylim[1]-self.ylim[0])

        x = Uniform(low=self.xlim[0], high=self.xlim[1]).sample(torch.Size([nsamp]))
        y = Uniform(low=self.ylim[0], high=self.ylim[1]).sample(torch.Size([nsamp]))
        D = torch.cat([x.view(-1, 1), y.view(-1, 1)], 1)

        _, ll = self.H.get_id(D)
        ltot = logsumexp_t(ll.view(-1))
        tsum = torch.exp(ltot)
        tot = (area/nsamp) * tsum

        print('Average score: %0.3f' % tot)
        self.assertTrue(np.abs(1 - tot) <= 1e-1)

    # TODO - more unit tests

if __name__ == '__main__':
    unittest.main()