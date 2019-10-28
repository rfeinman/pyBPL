import unittest
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform

from ..library import SpatialHist

class TestSpatialHist(unittest.TestCase):

    def setUp(self):
        # Parameters
        xlim = torch.tensor([-30, 20], dtype=torch.float)
        ylim = torch.tensor([-15, 15], dtype=torch.float)

        # Build the SpatialHist instance & sample data
        hist = SpatialHist(xlim, ylim)
        hist.initialize_unif()
        self.H = hist

        self.xlim = xlim
        self.ylim = ylim

    def test_sample(self):
        """
        Plot the sampled data next to the original data to verify that it
        looks correct.
        """
        n = 1000
        samples = self.H.sample(n)
        assert samples.shape == torch.Size([n,2])
        x_oob = (samples[:,0] < self.xlim[0]) | (samples[:,0] > self.xlim[1])
        y_oob = (samples[:,1] < self.ylim[0]) | (samples[:,1] > self.ylim[1])
        assert not torch.any(x_oob)
        assert not torch.any(y_oob)
        # TODO - more tests here?

    def test_score(self):
        """
        Plot the sampled data next to the original data to verify that it
        looks correct.
        """
        n = 1000
        x = np.random.uniform(low=self.xlim[0], high=self.xlim[1], size=(n,1))
        y = np.random.uniform(low=self.ylim[0], high=self.ylim[1], size=(n,1))
        data = np.hstack((x,y))
        data = torch.tensor(data, dtype=torch.float)
        ll = self.H.score(data)
        assert ll.shape == torch.Size([n])
        # TODO - more tests here?

    # TODO - more test functions

if __name__ == '__main__':
    unittest.main()