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
from ..concept.concept import Concept, ConceptToken


class CharacterToken(ConceptToken):
    def __init__(self, list_st, list_pos, affine, epsilon, blur_sigma, image):
        self.list_st = list_st
        self.list_pos = list_pos
        self.affine = affine
        self.epsilon = epsilon
        self.blur_sigma = blur_sigma
        self.image = image

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
        Sample a character token

        :return:
            token: [CharacterToken] character token
        """
        # sample the stroke tokens and the start positions
        list_st = []
        list_pos = []
        for s, r in zip(self.S, self.R):
            st = s.sample_token()
            pos = r.sample_position(prev_parts=list_st)
            list_st.append(st)
            list_pos.append(pos)

        # sample affine warp
        affine = self.sample_affine()

        # sample rendering parameters
        epsilon = self.sample_image_noise()
        blur_sigma = self.sample_image_blur()

        # get probability map of an image
        pimg, _ = rendering.apply_render(
            list_st, list_pos, affine, epsilon, blur_sigma, self.parameters
        )

        # sample the image
        image = sample_image(pimg)

        # create the character token
        token = ConceptToken(
            list_st, list_pos, affine, epsilon, blur_sigma, image
        )

        return token

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