import unittest
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

from pybpl.classes import SpatialHist

class TestSpatialHist(unittest.TestCase):

    def setUp(self):
        # Parameters
        nbin_per_side = 50
        prior_count = 0.1
        xlm = [-10, 10]
        ylm = [-10, 10]

        n = 1000  # number of training points in each slice

        # Shape of distribution
        mu1 = [-5, 0]
        mu2 = [0, 5]
        Sigma = np.eye(2)

        # Sample the data
        data1 = np.random.multivariate_normal(mu1, Sigma, n)
        data2 = np.random.multivariate_normal(mu2, Sigma, n)
        data = np.concatenate([data1, data2])

        # Build the SpatialHist instance & sample data
        self.H = SpatialHist(data, xlm, ylm, nbin_per_side, prior_count)
        self.syndata, _, _ = self.H.sample(1000)

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
        ll2 = np.sum(ll2)
        self.assertTrue(np.abs(ll - ll2) <= 1e-2)

    def test_validDensity(self):
        """
        Numerically check the normalizing constant of the density
        """
        nsamp = 10000
        area = (self.xlim[1]-self.xlim[0]) * (self.ylim[1]-self.ylim[0])

        x = np.random.uniform(low=self.xlim[0], high=self.xlim[1], size=nsamp)
        y = np.random.uniform(low=self.ylim[0], high=self.ylim[1], size=nsamp)
        D = np.transpose(np.vstack([x,y]))

        _, ll = self.H.get_id(D)
        ltot = logsumexp(ll.flatten())
        tsum = np.exp(ltot)
        tot = (area/nsamp) * tsum

        print('Average score: %0.3f' % tot)
        self.assertTrue(np.abs(1 - tot) <= 1e-1)

    # TODO - more unit tests

if __name__ == '__main__':
    unittest.main()