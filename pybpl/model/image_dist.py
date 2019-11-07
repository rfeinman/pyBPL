from __future__ import division, print_function
import torch
import torch.distributions as dist

from .. import rendering
from ..parameters import defaultps


class CharacterImageDist:
    """
    Defines the likelihood distribution P(Image | Token)
    """
    def __init__(self, lib):
        self.lib = lib
        self.default_ps = defaultps()

    def get_pimg(self, character_token):
        pimg, _ = rendering.apply_render(
            character_token.stroke_tokens, character_token.affine,
            character_token.epsilon, character_token.blur_sigma,
            self.default_ps)

        return pimg

    def sample_image(self, character_token):
        """
        Samples a binary image
        Note: Should only be called from Model

        Returns
        -------
        image : (H,W) tensor
            binary image sample
        """
        pimg = self.get_pimg(character_token)
        bern = dist.Bernoulli(pimg)
        image = bern.sample()

        return image

    def score_image(self, character_token, image):
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
        pimg = self.get_pimg(character_token)
        bern = dist.Bernoulli(pimg)
        ll = bern.log_prob(image)
        ll = torch.sum(ll)

        return ll
