from __future__ import division, print_function
import torch
import torch.distributions as dist

from .. import rendering


class ImageDist(object):
    """
    Likelihood Distribution
    """
    def __init__(self, lib):
        self.parameters = lib.parameters

    def sample_image(self, ctoken):
        """
        Samples a binary image
        Note: Should only be called from Model

        Returns
        -------
        image : (H,W) tensor
            binary image sample
        """
        pimg, _ = rendering.apply_render(
            ctoken.P, ctoken.affine, ctoken.epsilon, ctoken.blur_sigma,
            self.parameters
        )
        binom = dist.binomial.Binomial(1, pimg)
        image = binom.sample()

        return image

    def score_image(self, ctoken, image):
        """
        Compute the log-likelihood of a binary image
        Note: Should only be called from Model

        Parameters
        ----------
        image : (H,W) tensor
            binary image to score

        Returns
        -------
        ll : tensor
            scalar; log-likelihood of the image
        """
        pimg, _ = rendering.apply_render(
            ctoken.P, ctoken.affine, ctoken.epsilon, ctoken.blur_sigma,
            self.parameters
        )
        binom = dist.binomial.Binomial(1, pimg)
        ll = binom.log_prob(image)
        ll = torch.sum(ll)

        return ll