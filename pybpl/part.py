"""
Parts for sampling part tokens. Parts, together with relations between parts,
make up concepts.
"""
from __future__ import division, print_function
from abc import ABCMeta, abstractmethod
import torch
import torch.distributions as dist

from . import rendering


# --------------------- #
# parent 'Part' classes
# --------------------- #

class PartToken(object):
    """
    TODO
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass


class Part(object):
    """
    TODO
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def sample_token(self):
        pass

    @abstractmethod
    def score_token(self, token):
        pass


# ---------------------- #
# child 'Stroke' classes
# ---------------------- #


class StrokeToken(PartToken):
    """
    TODO

    Parameters
    ----------
    shapes : TODO
        TODO
    invscales : TODO
        TODO
    """
    def __init__(self, shapes, invscales):
        super(StrokeToken, self).__init__()
        self.shapes = shapes
        self.invscales = invscales
        self._position = None
        self.motor = None
        self.motor_spline = None

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        motor, motor_spline = rendering.vanilla_to_motor(
            self.shapes, self.invscales, value
        )
        self.motor = motor
        self.motor_spline = motor_spline
        self._position = value


class Stroke(Part):
    """
    A Stroke is a probabilistic program that can generate and score
    stroke tokens

    Parameters
    ----------
    TODO
    """
    def __init__(
            self, nsub, ids, shapes, invscales, lib
    ):
        super(Stroke, self).__init__()
        self.nsub = nsub
        self.ids = ids
        self.shapes = shapes
        self.invscales = invscales

        # distributions
        sigma_shape = lib.tokenvar['sigma_shape']
        sigma_invscale = lib.tokenvar['sigma_invscale']
        self.shapes_dist = dist.normal.Normal(shapes, sigma_shape)
        self.scales_dist = dist.normal.Normal(invscales, sigma_invscale)

    def sample_shapes_token(self):
        """
        TODO

        Returns
        -------
        shapes_token : TODO
            TODO
        """
        shapes_token = self.shapes_dist.sample()

        return shapes_token

    def score_shapes_token(self, shapes_token):
        """
        TODO

        Parameters
        ----------
        shapes_token : TODO
            TODO

        Returns
        -------
        ll : TODO
            TODO
        """
        # compute scores for every element in shapes_token
        ll = self.shapes_dist.log_prob(shapes_token)
        # sum scores
        ll = torch.sum(ll)

        return ll

    def sample_invscales_token(self):
        """
        TODO

        Returns
        -------
        invscales_token : TODO
            TODO
        """
        ll = torch.tensor(-float('inf'))
        while ll == -float('inf'):
            invscales_token = self.scales_dist.sample()
            ll = self.score_invscales_token(invscales_token)

        return invscales_token

    def score_invscales_token(self, invscales_token):
        """
        TODO

        Parameters
        ----------
        invscales_token : TODO
            TODO

        Returns
        -------
        ll : TODO
            TODO
        """
        # compute scores for every element in invscales_token
        ll = self.scales_dist.log_prob(invscales_token)

        # don't allow invscales that are negative
        out_of_bounds = invscales_token <= 0
        if out_of_bounds.any():
            ll = torch.tensor(-float('inf'))
            return ll

        # correction for positive only invscales
        p_below = self.scales_dist.cdf(0.)
        p_above = 1.- p_below
        ll = ll - torch.log(p_above)

        # sum scores
        ll = torch.sum(ll)

        return ll

    def sample_token(self):
        """
        TODO

        Parameters
        ----------
        position : TODO
            TODO

        Returns
        -------
        token : TODO
            TODO
        """
        shapes = self.sample_shapes_token()
        invscales = self.sample_invscales_token()
        token = StrokeToken(shapes, invscales)

        return token

    def score_token(self, token):
        """
        TODO

        Parameters
        ----------
        token

        Returns
        -------

        """
        ll = 0.
        ll = ll + self.score_shapes_token(token.shapes)
        ll = ll + self.score_invscales_token(token.invscales)

        return ll