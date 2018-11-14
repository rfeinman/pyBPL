from __future__ import division, print_function
from abc import ABCMeta, abstractmethod
import torch
import torch.distributions as dist

from .. import rendering
from ..parameters import defaultps


class ConceptImageDist(object):
    __metaclass__ = ABCMeta

    def __init__(self, lib):
        self.lib = lib

    @abstractmethod
    def sample_image(self, ctoken):
        pass

    @abstractmethod
    def score_image(self, ctoken, image):
        pass

class CharacterImageDist(ConceptImageDist):
    """
    Defines the likelihood distribution P(Image | Token)
    """
    def __init__(self, lib):
        super(CharacterImageDist, self).__init__(lib)
        self.default_ps = defaultps()

    def get_pimg(self, ctoken):
        pimg, _ = rendering.apply_render(
            ctoken.part_tokens, ctoken.affine, ctoken.epsilon,
            ctoken.blur_sigma, self.default_ps
        )

        return pimg

    def sample_image(self, ctoken):
        """
        Samples a binary image
        Note: Should only be called from Model

        Returns
        -------
        image : (H,W) tensor
            binary image sample
        """
        pimg = self.get_pimg(ctoken)
        bern = dist.Bernoulli(pimg)
        image = bern.sample()

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
        pimg = self.get_pimg(ctoken)
        bern = dist.Bernoulli(pimg)
        ll = bern.log_prob(image)
        ll = torch.sum(ll)

        return ll