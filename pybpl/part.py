"""
Parts for sampling part tokens. Parts, together with relations between parts,
make up concepts.
"""
from __future__ import division, print_function
from abc import ABCMeta, abstractmethod
import torch
import torch.distributions as dist

from .token import StrokeToken



class Part(object):
    """
    Part class TODO
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def sample_token(self, position_token):
        pass



class Stroke(Part):
    """
    A Stroke is a probabilistic program that can generate and score
    stroke tokens

    Parameters
    ----------
    ids : TODO
        TODO
    shapes_type : TODO
        TODO
    invscales_type : TODO
        TODO
    lib : Library
        TODO
    """
    def __init__(
            self, ids, shapes_type, invscales_type, sigma_shape, sigma_invscale
    ):
        # parent init
        super(Stroke, self).__init__()

        # type-level parameters
        self.ids = ids
        self.shapes_type = shapes_type
        self.invscales_type = invscales_type

        # distributions
        self.shapes_dist = dist.normal.Normal(shapes_type, sigma_shape)
        self.scales_dist = dist.normal.Normal(invscales_type, sigma_invscale)

    @property
    def nsub(self):
        """
        The number of sub-strokes
        """
        return torch.tensor(len(self.ids))

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

    def sample_token(self, position_token):
        """
        TODO

        Parameters
        ----------
        position_token : TODO
            TODO

        Returns
        -------
        token : TODO
            TODO
        """
        shapes_token = self.sample_shapes_token()
        invscales_token = self.sample_invscales_token()
        token = StrokeToken(shapes_token, invscales_token, position_token)

        return token