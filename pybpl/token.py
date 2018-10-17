"""
Token classes for representing & storing concept tokens.
"""
from __future__ import division, print_function
from abc import ABCMeta, abstractmethod

from . import rendering



class PartToken(object):
    """
    PartToken class TODO
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass


class StrokeToken(PartToken):
    """
    StrokeToken class TODO

    Parameters
    ----------
    shapes : TODO
        TODO
    invscales : TODO
        TODO
    position : TODO
        TODO
    """
    def __init__(self, shapes, invscales, position):
        super(StrokeToken, self).__init__()
        self.shapes = shapes
        self.invscales = invscales
        self.position = position
        self.motor, self.motor_spline = rendering.vanilla_to_motor(
            self.shapes, self.invscales, self.position
        )


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