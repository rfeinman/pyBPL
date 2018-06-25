"""
Stroke class definition.
"""
from __future__ import print_function, division
import numpy as np
import torch
import torch.distributions as dist

from .. import rendering
from ..concept.part import Part, PartToken


class StrokeToken(PartToken):
    def __init__(self, shapes, invscales):
        PartToken.__init__(self)
        self.shapes = shapes
        self.invscales = invscales

    def motor(self, position):
        """
        Compute the [x,y,t] trajectory of this stroke
        """
        motor, _ = rendering.vanilla_to_motor(
            self.shapes, self.invscales, position
        )

        return motor

    def motor_spline(self, position):
        """
        Compute the spline trajectory of this stroke
        """
        _, motor_spline = rendering.vanilla_to_motor(
            self.shapes, self.invscales, position
        )

        raise NotImplementedError

class Stroke(Part):
    """
    A Stroke is a program that can generate and score new stroke tokens
    """
    def __init__(
            self, ids, shapes_type, invscales_type, sigma_shape, sigma_invscale
    ):
        """
        Initialize the Stroke class instance.

        :param ids:
        :param shapes_type:
        :param invscales_type:
        :param lib:
        """
        # parent init
        Part.__init__(self)

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
        Get the number of sub-strokes
        """
        return len(self.ids)

    def sample_shapes_token(self):
        shapes_token = self.shapes_dist.sample()

        return shapes_token

    def score_shapes_token(self, shapes_token):
        # compute scores for every element in shapes_token
        ll = self.shapes_dist.log_prob(shapes_token)
        # sum scores
        ll = torch.sum(ll)

        return ll

    def sample_invscales_token(self):
        ll = torch.tensor(-np.inf)
        while np.isinf(ll):
            invscales_token = self.scales_dist.sample()
            ll = self.score_invscales_token(invscales_token)

        return invscales_token

    def score_invscales_token(self, invscales_token):
        # compute scores for every element in invscales_token
        ll = self.scales_dist.log_prob(invscales_token)

        # don't allow invscales that are negative
        out_of_bounds = invscales_token <= 0
        if out_of_bounds.any():
            ll = torch.tensor(-np.inf)
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

        :return:
        """
        shapes_token = self.sample_shapes_token()
        invscales_token = self.sample_invscales_token()
        token = StrokeToken(shapes_token, invscales_token)

        return token