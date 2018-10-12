"""
Module for concepts.

A concept is a meta-class. Concepts are probabilistic programs that
sample concept tokens. A concept contains a sequence of parts and a sequence of
relations to connect each part to previous parts.

This class is inherited from by child classes for specific types of concepts.
One example of such child class is Character.

Classes:
    Concept: ...
    ConceptToken: ...
    ConceptTypeDist: ...
Functions:
    ...
"""
from __future__ import division, print_function
from abc import ABCMeta, abstractmethod
import warnings
import torch
import torch.distributions as dist

from . import rendering
from .parameters import defaultps
from .library import Library
from .part import Part, Stroke
from .relation import Relation
from .token import CharacterToken


class Concept(object):
    __metaclass__ = ABCMeta

    def __init__(self, P, R):
        assert isinstance(P, list)
        assert isinstance(R, list)
        assert len(P) == len(R)
        assert len(P) > 0
        for p, r in zip(P, R):
            assert isinstance(p, Part)
            assert isinstance(r, Relation)
        self.P = P
        self.R = R

    @property
    def k(self):
        # the number of parts
        return len(self.P)

    @abstractmethod
    def render_part(self, part_token, part_location):
        pass

    @abstractmethod
    def sample_token(self):
        part_tokens = []
        for part, rel in zip(self.P, self.R):
            part_location = rel.sample_position(part_tokens)
            part_token = part.sample_token(part_location)
            part_tokens.append(part_token)

        return part_tokens



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

    def sample_token(self):
        """
        Sample a character token

        :return:
            token: [CharacterToken] character token
        """
        stroke_tokens = super(Character, self).sample_token()

        # sample affine warp
        affine = self.sample_affine() # (4,) tensor

        # sample rendering parameters
        epsilon = self.sample_image_noise()
        blur_sigma = self.sample_image_blur()

        # get probability map of an image
        pimg, _ = rendering.apply_render(
            stroke_tokens, affine, epsilon, blur_sigma, self.parameters
        )

        # sample the image
        image = sample_image(pimg)

        # create the character token
        token = CharacterToken(
            stroke_tokens, affine, epsilon, blur_sigma, image
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
