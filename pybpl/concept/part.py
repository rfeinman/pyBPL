from __future__ import division, print_function
from abc import ABCMeta, abstractmethod


class RenderedPart(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

class PartToken(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

class Part(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def sample_token(self):
        pass