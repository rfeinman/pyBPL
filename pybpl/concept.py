"""
Concepts are probabilistic programs that sample concept tokens. A concept
contains a sequence of parts and a sequence of relations to connect each
part to previous parts. The Concept class is an abstract class, and must be
inherited from by derivative classes. It defines the general overarching
structure that child classes need to adhere to.

One example of such child class is the Character class. This contains the
implementation of the Omniglot BPL use case. Parts are Strokes and ... TODO
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
from .token import ConceptToken, CharacterToken


class Concept(object):
    """
    An abstract base class for concepts. A concept is a probabilistic program
    that samples Concept tokens. Concepts are made up of parts and relations.

    Parameters
    ----------
    P : list of Part
        TODO
    R : list of Relation
        TODO
    """
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
        """
        The number of parts
        """
        return torch.tensor(len(self.P))

    @abstractmethod
    def sample_token(self):
        part_tokens = []
        for part, rel in zip(self.P, self.R):
            part_location = rel.sample_position(part_tokens)
            part_token = part.sample_token(part_location)
            part_tokens.append(part_token)
        token = ConceptToken(part_tokens)

        return token



class Character(Concept):
    """
    A Character is a probabilistic program that samples Character tokens.
    Parts are strokes, and relations are either [independent, attach,
    attach-along].

    Parameters
    ----------
    S : list of Stroke
        TODO
    R : list of Relation
        TODO
    lib : Library
        TODO
    """
    def __init__(self, S, R, lib):
        for s in S:
            assert isinstance(s, Stroke)
        assert isinstance(lib, Library)
        super(Character, self).__init__(P=S, R=R)
        self.lib = lib
        self.parameters = defaultps()

    def sample_token(self):
        """
        Sample a character token

        Returns
        -------
        token : CharacterToken
            character token
        """
        concept_token = super(Character, self).sample_token()
        stroke_tokens = concept_token.part_tokens

        # sample affine warp
        affine = self.sample_affine() # (4,) tensor

        # sample rendering parameters
        epsilon = self.sample_image_noise()
        blur_sigma = self.sample_image_blur()

        # create the character token
        token = CharacterToken(stroke_tokens, affine, epsilon, blur_sigma)

        # get probability map of an image
        pimg, _ = rendering.apply_render(
            token, self.parameters
        )

        # sample the image
        image = sample_image(pimg)

        return token, image

    def sample_affine(self):
        """
        TODO

        Returns
        -------
        affine : TODO
            affine transformation
        """
        warnings.warn('skipping affine warp for now.')
        affine = None

        return affine

    def sample_image_noise(self):
        """
        TODO

        Returns
        -------
        epsilon : float
            scalar; image noise quantity
        """
        #epsilon = CPD.sample_image_noise(self.parameters)
        warnings.warn('using fixed image noise for now.')
        # set rendering parameters to minimum noise
        epsilon = self.parameters.min_epsilon

        return epsilon

    def sample_image_blur(self):
        """
        TODO

        Returns
        -------
        blur_sigma: float
            scalar; image blur quantity
        """
        #blur_sigma = CPD.sample_image_blur(self.parameters)
        warnings.warn('using fixed image blur for now.')
        # set rendering parameters to minimum noise
        blur_sigma = self.parameters.min_blur_sigma

        return blur_sigma

def sample_image(pimg):
    """
    TODO

    Parameters
    ----------
    pimg : tensor
        image probability map

    Returns
    -------
    image : tensor
        binary image
    """
    binom = dist.binomial.Binomial(1, pimg)
    image = binom.sample()

    return image

def score_image(image, pimg):
    """
    TODO

    Parameters
    ----------
    image : tensor
        binary image
    pimg : tensor
        image probability map

    Returns
    -------
    ll : tensor
        scalar; log-likelihood of the image
    """
    binom = dist.binomial.Binomial(1, pimg)
    ll = binom.log_prob(image)

    return ll
