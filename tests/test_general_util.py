from __future__ import division, print_function
import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from pybpl.general_util import ind2sub, fspecial, imfilter

class TestGeneralUtil(unittest.TestCase):

    def testFspecial(self):
        # make sure even 'hsize' parameter raises exception
        with self.assertRaises(AssertionError):
            fspecial(10, 2)
        # initialize the gaussian kernel
        kernel = fspecial(11, 2, ftype='gaussian')
        # make sure it's a tensor
        self.assertIsInstance(kernel, torch.Tensor)
        # make sure it has the right shape
        self.assertTrue(kernel.shape == torch.Size([11,11]))
        # make sure it has its mean in the center
        mean_i = torch.argmax(kernel)
        mean_yi, mean_xi = ind2sub(kernel.shape, mean_i)
        self.assertEqual(mean_yi, torch.tensor(5))
        self.assertEqual(mean_xi, torch.tensor(5))
        # show the filter
        plt.figure(figsize=(1, 1))
        plt.imshow(kernel.numpy())
        plt.axis('off')
        plt.show()

    def testImfilter(self):
        # load the test image
        im_path = os.path.join(os.environ['PYBPL_DIR'], 'tests/Curtis.jpg')
        im = plt.imread(im_path)
        # convert to black-and-white, range 0-1
        im = im[:,:,0]
        im = im / 255.
        # convert to torch Tensor
        im = torch.tensor(im, dtype=torch.float32)
        plt.figure(figsize=(10,8))
        plt.subplot(2,2,1)
        plt.imshow(1-im.numpy(), cmap='Greys', vmin=0, vmax=1)
        plt.axis('off')
        plt.title('Original')
        # try Guassian blur with a few different sigma values
        sigmas = [1., 2., 3.]
        for i, sigma in enumerate(sigmas):
            kernel = fspecial(11, sigma, ftype='gaussian')
            im_blurred = imfilter(im, kernel, mode='conv')
            plt.subplot(2,2,i+2)
            plt.imshow(1 - im_blurred.numpy(), cmap='Greys', vmin=0, vmax=1)
            plt.axis('off')
            plt.title('Blur sigma = %0.1f' % sigma)
        plt.show()


if __name__ == "__main__":
    unittest.main()