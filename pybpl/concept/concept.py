from __future__ import division, print_function
from abc import ABCMeta, abstractmethod


class ConceptToken(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        return

class Concept(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        return

    @abstractmethod
    def sample_token(self):
        return