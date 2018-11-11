"""
Parts for sampling part tokens. Parts, together with relations between parts,
make up concepts.
"""
from __future__ import division, print_function
from abc import ABCMeta, abstractmethod
import torch

from . import rendering


class PartType(object):
    """
    An abstract base class for parts. Holds all type-level parameters of the
    part.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def parameters(self):
        """
        return list of parameters
        """
        pass

    @abstractmethod
    def lbs(self, eps=1e-4):
        """
        return list of lower bounds for parameters
        """

    @abstractmethod
    def ubs(self, eps=1e-4):
        """
        return list of upper bounds for parameters
        """

    def train(self):
        """
        makes params require grad
        TODO
        """
        pass

    def eval(self):
        """
        makes params require no grad
        TODO
        """
        pass

    def to(self, device):
        """
        moves parameters to device
        TODO
        """
        pass


class StrokeType(PartType):
    """
    Holds all type-level parameters of the stroke.

    Parameters
    ----------
    nsub : tensor
        scalar; number of sub-strokes
    ids : (nsub,) tensor
        sub-stroke ID sequence
    shapes : (ncpt, 2, nsub) tensor
        shapes types
    invscales : (nsub,) tensor
        invscales types
    """
    def __init__(self, nsub, ids, shapes, invscales):
        super(StrokeType, self).__init__()
        self.nsub = nsub
        self.ids = ids
        self.shapes = shapes
        self.invscales = invscales

    def parameters(self):
        """
        Returns a list of parameters that can be optimized via gradient descent.

        Returns
        -------
        parameters : list
            optimizable parameters
        """
        parameters = [self.shapes, self.invscales]

        return parameters

    def lbs(self, eps=1e-4):
        """
        Returns a list of lower bounds for each of the optimizable parameters.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        lbs : list
            lower bound for each parameter
        """
        lbs = [None, torch.full(self.invscales.shape, eps)]

        return lbs

    def ubs(self, eps=1e-4):
        """
        Returns a list of upper bounds for each of the optimizable parameters.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        ubs : list
            upper bound for each parameter
        """
        ubs = [None, None]

        return ubs


class PartToken(object):
    """
    An abstract base class for part tokens. Holds all token-level parameters
    of the part.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def parameters(self):
        """
        return list of parameters
        """
        pass

    @abstractmethod
    def lbs(self, eps=1e-4):
        """
        return list of lower bounds for parameters
        """

    @abstractmethod
    def ubs(self, eps=1e-4):
        """
        return list of upper bounds for parameters
        """

    def train(self):
        """
        makes params require grad
        TODO
        """
        pass

    def eval(self):
        """
        makes params require no grad
        TODO
        """
        pass

    def to(self, device):
        """
        moves parameters to device
        TODO
        """
        pass


class StrokeToken(PartToken):
    """
    Stroke tokens hold all token-level parameters of the stroke.

    Parameters
    ----------
    shapes : (ncpt, 2, nsub) tensor
        shapes tokens
    invscales : (nsub,) tensor
        invscales tokens
    """
    def __init__(self, shapes, invscales):
        super(StrokeToken, self).__init__()
        self.shapes = shapes
        self.invscales = invscales
        self.position = None

    @property
    def motor(self):
        """
        TODO
        """
        assert self.position is not None
        motor, _ = rendering.vanilla_to_motor(
            self.shapes, self.invscales, self.position
        )

        return motor

    @property
    def motor_spline(self):
        """
        TODO
        """
        assert self.position is not None
        _, motor_spline = rendering.vanilla_to_motor(
            self.shapes, self.invscales, self.position
        )

        return motor_spline

    def parameters(self):
        """
        Returns a list of parameters that can be optimized via gradient descent.

        Returns
        -------
        parameters : list
            optimizable parameters
        """
        parameters = [self.shapes, self.invscales]

        return parameters

    def lbs(self, eps=1e-4):
        """
        Returns a list of lower bounds for each of the optimizable parameters.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        lbs : list
            lower bound for each parameter
        """
        lbs = [None, torch.full(self.invscales.shape, eps)]

        return lbs

    def ubs(self, eps=1e-4):
        """
        Returns a list of upper bounds for each of the optimizable parameters.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        ubs : list
            upper bound for each parameter
        """
        ubs = [None, None]

        return ubs


