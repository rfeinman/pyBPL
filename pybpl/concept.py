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
from .part import PartToken, StrokeToken, Stroke
from .ctd import ConceptType



class ConceptToken(object):
    """
    Abstract base class for concept tokens

    Parameters
    ----------
    part_tokens : list of PartToken
        TODO
    """
    __metaclass__ = ABCMeta

    def __init__(self, part_tokens):
        for pt in part_tokens:
            assert isinstance(pt, PartToken)
        self.part_tokens = part_tokens


class Concept(object):
    """
    An abstract base class for concepts. A concept is a probabilistic program
    that samples Concept tokens. Concepts are made up of parts and relations.

    Parameters
    ----------
    ctype : ConceptType
        concept type
    """
    __metaclass__ = ABCMeta

    def __init__(self, ctype):
        assert isinstance(ctype, ConceptType)
        self.ctype = ctype

    @property
    def k(self):
        """
        The number of parts
        """
        return self.ctype.k

    @abstractmethod
    def sample_token(self):
        part_tokens = []
        for part, rel in zip(self.ctype.P, self.ctype.R):
            part_location = rel.sample_position(part_tokens)
            part_token = part.sample_token(part_location)
            part_tokens.append(part_token)
        token = ConceptToken(part_tokens)

        return token


class CharacterToken(ConceptToken):
    """
    Character token stores both the type- and token-level parameters of a
    character sample

    Parameters
    ----------
    stroke_tokens : TODO
        TODO
    affine : TODO
        TODO
    epsilon : TODO
        TODO
    blur_sigma : TODO
        TODO
    """
    def __init__(self, stroke_tokens, affine, epsilon, blur_sigma):
        super(CharacterToken, self).__init__(stroke_tokens)
        for token in stroke_tokens:
            assert isinstance(token, StrokeToken)
        self.stroke_tokens = stroke_tokens
        self.affine = affine
        self.epsilon = epsilon
        self.blur_sigma = blur_sigma


class Character(Concept):
    """
    A Character is a probabilistic program that samples Character tokens.
    Parts are strokes, and relations are either [independent, attach,
    attach-along].

    Parameters
    ----------
    ctype : ConceptType
        concept type
    lib : Library
        library instance
    """
    def __init__(self, ctype, lib):
        super(Character, self).__init__(ctype)
        for p in ctype.P:
            assert isinstance(p, Stroke)
        assert isinstance(lib, Library)
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
