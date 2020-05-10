from abc import ABCMeta, abstractmethod
import torch
import torch.distributions as dist

from ..rendering import render_image, apply_warp
from ..parameters import Parameters


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
        self.ps = Parameters()

    def get_pimg(self, ctoken):
        # get motor for each part
        motor = [p.motor for p in ctoken.part_tokens] # list of (nsub, ncpt, 2)
        # apply affine transformation if needed
        if ctoken.affine is not None:
            motor = apply_warp(motor, ctoken.affine)
        motor_flat = torch.cat(motor) # (nsub_total, ncpt, 2)
        pimg, _ = render_image(
            motor_flat, ctoken.epsilon, ctoken.blur_sigma, self.ps)

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