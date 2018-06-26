from __future__ import division, print_function
from abc import ABCMeta, abstractmethod

from .part import Part
from .relation import Relation

class ConceptToken(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

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
        rendered_parts = []
        for part, rel in zip(self.P, self.R):
            part_token = part.sample_token()
            part_location = rel.sample_position(rendered_parts)
            rendered_part = self.render_part(part_token, part_location)
            rendered_parts.append(rendered_part)

        return rendered_parts

