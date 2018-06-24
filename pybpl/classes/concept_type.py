"""
A concept type. This serves as a template for a motor program.
"""
from __future__ import division, print_function
from .stroke import StrokeType
from .relations import Relation

class ConceptType(object):
    def __init__(self, S, R):
        for stype in S:
            assert isinstance(stype, StrokeType)
        for rtype in R:
            assert isinstance(rtype, Relation)
        assert len(S) == len(R)
        # the list of stroke types
        self.S = S
        # the list of relations
        self.R = R

    @property
    def ns(self):
        # get number of strokes
        return len(self.S)