"""
Token classes for representing & storing concept tokens.
"""
from __future__ import division, print_function
from abc import ABCMeta, abstractmethod

from .part import StrokeToken

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
        self.part_tokens = part_tokens

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
    image : TODO
        TODO
    """
    def __init__(self, stroke_tokens, affine, epsilon, blur_sigma, image):
        super(CharacterToken, self).__init__()
        for token in stroke_tokens:
            assert isinstance(token, StrokeToken)
        self.stroke_tokens = stroke_tokens
        self.affine = affine
        self.epsilon = epsilon
        self.blur_sigma = blur_sigma
        self.image = image