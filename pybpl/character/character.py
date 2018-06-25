"""
Character class definition
"""
from __future__ import print_function, division
import warnings
import torch
import torch.distributions as dist

from .stroke import Stroke
from .parameters import defaultps
from ..concept.relation import Relation
from ..library.library import Library
from .. import CPD
from .. import rendering
from ..concept.concept import Concept


class Character(Concept):
    """
    TODO
    """
    def __init__(self, S, R, lib):
        """
        Constructor

        :param S: [list of Stroke] TODO
        :param R: [list of Relation] TODO
        :param lib: [Library] TODO
        """
        assert len(S) == len(R)
        assert len(S) > 0
        for s, r in zip(S, R):
            assert isinstance(s, Stroke)
            assert isinstance(r, Relation)
        assert isinstance(lib, Library)
        Concept.__init__(self)
        self.S = S
        self.R = R
        self.lib = lib
        self.parameters = defaultps()

    @property
    def ns(self):
        # get number of strokes
        return len(self.S)

    def sample_token(self):
        """
        Sample a token from the motor program

        :return:
            image: [(m,n) tensor] token (image)
        """
        # sample the stroke and relation tokens
        list_st = []
        list_rt = []
        for s, r in zip(self.ctype.S, self.ctype.R):
            s_token = s.sample_token()
            r_token = r.sample_token(prev_parts=list_st)
            list_st.append(s_token)
            list_rt.append(r_token)

        # sample affine warp
        affine = self.sample_affine()

        # sample rendering parameters
        epsilon = self.sample_image_noise()
        blur_sigma = self.sample_image_blur()

        # get probability map of an image
        pimg, _ = rendering.apply_render(
            list_st, list_rt, affine, epsilon, blur_sigma, self.parameters
        )

        # sample the image
        image = sample_image(pimg)

        return image

    def sample_affine(self):
        raise NotImplementedError
        return None

    def sample_image_noise(self):
        #epsilon = CPD.sample_image_noise(self.parameters)
        warnings.warn('using fixed image noise for now...')
        # set rendering parameters to minimum noise
        epsilon = self.parameters.min_epsilon

        return epsilon

    def sample_image_blur(self):
        #blur_sigma = CPD.sample_image_blur(self.parameters)
        warnings.warn('using fixed image blur for now...')
        # set rendering parameters to minimum noise
        blur_sigma = self.parameters.min_blur_sigma

        return blur_sigma

def sample_image(pimg):
    binom = dist.binomial.Binomial(1, pimg)
    image = binom.sample()

    return image

def score_image(image, pimg):
    binom = dist.binomial.Binomial(1, pimg)
    ll = binom.log_prob(image)

    return ll