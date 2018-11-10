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
    An abstract base class for part tokens. Holds all token-level parameters
    of the part.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def optimizable_parameters(self, eps=1e-4):
        pass


class Part(object):
    """
    An abstract base class for parts. Holds all type-level parameters of the
    part.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def optimizable_parameters(self, eps=1e-4):
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

    def optimizable_parameters(self, eps=1e-4):
        """
        Returns a list of parameters that can be optimized via gradient descent.
        Includes lists of lower and upper bounds, with one per parameter.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        params : list
            optimizable parameters
        lbs : list
            lower bound for each parameter
        ubs : list
            upper bound for each parameter
                """
        params = [self.shapes, self.invscales]
        lbs = [None, torch.full(self.invscales.shape, eps)]
        ubs = [None, None]

        return params, lbs, ubs

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


class Stroke(Part):
    """
    A Stroke is a probabilistic program that can generate and score
    stroke tokens

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
    lib : Library
        library instance
    """
    def __init__(self, nsub, ids, shapes, invscales, lib):
        super(Stroke, self).__init__()
        self.nsub = nsub
        self.ids = ids
        self.shapes = shapes
        self.invscales = invscales

        # distributions
        self.sigma_shape = lib.tokenvar['sigma_shape']
        self.sigma_invscale = lib.tokenvar['sigma_invscale']

    def optimizable_parameters(self, eps=1e-4):
        """
        Returns a list of parameters that can be optimized via gradient descent.
        Includes lists of lower and upper bounds, with one per parameter.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        params : list
            optimizable parameters
        lbs : list
            lower bound for each parameter
        ubs : list
            upper bound for each parameter
        """
        params = [self.shapes, self.invscales]
        lbs = [None, torch.full(self.invscales.shape, eps)]
        ubs = [None, None]

        return params, lbs, ubs

    def sample_shapes_token(self):
        """
        Sample a token of each sub-stroke's shapes

        Returns
        -------
        shapes_token : (ncpt, 2, nsub) tensor
            sampled shapes token
        """
        shapes_dist = dist.normal.Normal(self.shapes, self.sigma_shape)
        # sample shapes token
        shapes_token = shapes_dist.sample()

        return shapes_token

    def score_shapes_token(self, shapes_token):
        """
        Compute the log-likelihood of each sub-strokes's shapes

        Parameters
        ----------
        shapes_token : (ncpt, 2, nsub) tensor
            shapes tokens to score

        Returns
        -------
        ll : (nsub,) tensor
            vector of log-likelihood scores
        """
        shapes_dist = dist.normal.Normal(self.shapes, self.sigma_shape)
        # compute scores for every element in shapes_token
        ll = shapes_dist.log_prob(shapes_token)

        return ll

    def sample_invscales_token(self):
        """
        Sample a token of each sub-stroke's scale

        Returns
        -------
        invscales_token : (nsub,) tensor
            sampled scales tokens
        """
        scales_dist = dist.normal.Normal(self.invscales, self.sigma_invscale)
        while True:
            invscales_token = scales_dist.sample()
            ll = self.score_invscales_token(invscales_token)
            if not torch.any(ll == -float('inf')):
                break

        return invscales_token

    def score_invscales_token(self, invscales_token):
        """
        Compute the log-likelihood of each sub-stroke's scale

        Parameters
        ----------
        invscales_token : (nsub,) tensor
            scales tokens to score

        Returns
        -------
        ll : (nsub,) tensor
            vector of log-likelihood scores
        """
        scales_dist = dist.normal.Normal(self.invscales, self.sigma_invscale)
        # compute scores for every element in invscales_token
        ll = scales_dist.log_prob(invscales_token)

        # correction for positive only invscales
        p_below = scales_dist.cdf(0.)
        p_above = 1. - p_below
        ll = ll - torch.log(p_above)

        # don't allow invscales that are negative
        out_of_bounds = invscales_token <= 0
        ll[out_of_bounds] = -float('inf')

        return ll

    def sample_token(self):
        """
        Sample a stroke token

        Returns
        -------
        token : StrokeToken
            stroke token sample
        """
        shapes = self.sample_shapes_token()
        invscales = self.sample_invscales_token()
        token = StrokeToken(shapes, invscales)

        return token

    def score_token(self, token):
        """
        Compute the log-likelihood of a stroke token

        Parameters
        ----------
        token : StrokeToken
            stroke token to score

        Returns
        -------
        ll : tensor
            scalar; log-likelihood of the stroke token
        """
        shapes_scores = self.score_shapes_token(token.shapes)
        invscales_scores = self.score_invscales_token(token.invscales)
        ll = torch.sum(shapes_scores) + torch.sum(invscales_scores)

        return ll