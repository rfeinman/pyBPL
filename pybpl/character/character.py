"""
Character class definition
"""
from __future__ import print_function, division
import warnings
import torch
import torch.distributions as dist

from .stroke import Stroke, RenderedStroke
from .parameters import defaultps
from ..library.library import Library
from .. import CPD
from .. import rendering
from ..concept.concept import Concept, ConceptToken


class CharacterToken(ConceptToken):
    def __init__(self, rendered_parts, affine, epsilon, blur_sigma, image):
        super(CharacterToken, self).__init__()
        self.rendered_parts = rendered_parts
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
        for s in S:
            assert isinstance(s, Stroke)
        assert isinstance(lib, Library)
        super(Character, self).__init__(P=S, R=R)
        self.lib = lib
        self.parameters = defaultps()

    def render_part(self, part_token, part_location):
        motor, motor_spline = rendering.vanilla_to_motor(
            part_token.shapes, part_token.invscales, part_location
        )
        rendered_stroke = RenderedStroke(motor, motor_spline)

        return rendered_stroke

    def sample_token(self):
        """
        Sample a character token

        :return:
            token: [CharacterToken] character token
        """
        rendered_strokes = super(Character, self).sample_token()

        # sample affine warp
        affine = self.sample_affine() # (4,) tensor

        # sample rendering parameters
        epsilon = self.sample_image_noise()
        blur_sigma = self.sample_image_blur()

        # get probability map of an image
        pimg, _ = rendering.apply_render(
            rendered_strokes, affine, epsilon, blur_sigma, self.parameters
        )

        # sample the image
        image = sample_image(pimg)

        # create the character token
        token = CharacterToken(
            rendered_strokes, affine, epsilon, blur_sigma, image
        )

        return token

    def sample_affine(self):
        warnings.warn('skipping affine warp for now.')
        affine = None

        return affine

    def sample_image_noise(self):
        #epsilon = CPD.sample_image_noise(self.parameters)
        warnings.warn('using fixed image noise for now.')
        # set rendering parameters to minimum noise
        epsilon = self.parameters.min_epsilon

        return epsilon

    def sample_image_blur(self):
        #blur_sigma = CPD.sample_image_blur(self.parameters)
        warnings.warn('using fixed image blur for now.')
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