import unittest
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from ..library import SpatialModel


class TestSpatialModel(unittest.TestCase):

    def setUp(self):
        # Parameters
        clump_id = 1
        nbin_per_side = 15
        prior_count = 4.
        xlm = [-10, 10]
        ylm = [-10, 10]

        n_train = int(1e5) # number of training points in each slice
        n_train_last = 100 # number of datapoints from last slice
        n_test = 1000 # number of test points

        # Shape of the distributions
        mu1 = torch.tensor([-5., -5.])
        mu2 = torch.tensor([0., 0.])
        mu3 = torch.tensor([5., 5.])
        Sigma = torch.eye(2)

        # Sample the data
        data1 = MultivariateNormal(mu1, Sigma).sample(torch.Size([n_train]))
        data2 = MultivariateNormal(mu2, Sigma).sample(torch.Size([n_train]))
        data3 = MultivariateNormal(mu3, Sigma).sample(torch.Size([n_train]))
        data4 = MultivariateNormal(mu3, Sigma).sample(torch.Size([n_train_last]))

        # Dataset to fit the histogram model to
        data_train = torch.cat([data1, data2, data3, data4])
        indx_train = torch.cat(
            [torch.zeros(n_train), torch.ones(n_train),
             2*torch.ones(n_train), 3*torch.ones(n_train_last)]
        )

        # Build the SpatialModel instance
        self.GT = SpatialModel(
            data_train, indx_train, clump_id, xlm, ylm,
            nbin_per_side, prior_count
        )

        self.clump_id = clump_id
        self.nbin_per_side = nbin_per_side
        self.prior_count = prior_count
        self.data_train = data_train
        self.indx_train = indx_train
        self.n_test = n_test

    def test_plot(self):
        """
        Test visualization of learned position models
        :return:
        """
        self.GT.plot()
        # TODO - write assertions

    def test_selectionPosition(self):
        n = self.n_test
        # Create validation set
        indx_test = torch.cat(
            [torch.zeros(n), torch.ones(n), 2*torch.ones(n), 3*torch.ones(n)]
        )
        data_test = self.GT.sample(indx_test)
        print('Ground truth parameters:');
        print('  number of bins: %i' % self.nbin_per_side)
        print('  additive count: %f' % self.prior_count)
        print('  clump at: %i' % self.clump_id)
        # TODO - write assertions

    # TODO - more unit tests

if __name__ == '__main__':
    unittest.main()