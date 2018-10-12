"""
A module for tokens.

Classes:
    ...
Functions:
    ...
"""
from __future__ import division, print_function
from abc import ABCMeta, abstractmethod

from .part import StrokeToken

class ConceptToken(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

class CharacterToken(ConceptToken):
    def __init__(self, stroke_tokens, affine, epsilon, blur_sigma, image):
        super(CharacterToken, self).__init__()
        for token in stroke_tokens:
            assert isinstance(token, StrokeToken)
        self.stroke_tokens = stroke_tokens
        self.affine = affine
        self.epsilon = epsilon
        self.blur_sigma = blur_sigma
        self.image = image