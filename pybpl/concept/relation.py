"""
Relation class definitions
"""
from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import numpy as np
import torch
import torch.distributions as dist

from .part import PartToken
from ..splines import bspline_eval, bspline_gen_s

types_allowed = ['unihist', 'start', 'end', 'mid']


class RelationToken(object):
    def __init__(self, position):
        self.position = position


class Relation(object):
    __metaclass__ = ABCMeta

    def __init__(self, rtype, nprev, sigma_x, sigma_y):
        # make sure type is valid
        assert rtype in types_allowed
        self.type = rtype
        self.nprev = nprev

        # build the position distribution
        Cov = torch.eye(2)
        Cov[0,0] = sigma_x
        Cov[1,1] = sigma_y
        self.pos_dist = dist.multivariate_normal.MultivariateNormal(
            torch.zeros(2), Cov
        )

    def sample_position_token(self, prev_parts):
        base = self.get_attach_point(prev_parts)
        assert base.shape == torch.Size([2])
        pos = base + self.pos_dist.sample()

        return pos

    def sample_token(self, prev_parts):
        for part in prev_parts:
            assert isinstance(part, PartToken)
        pos = self.sample_position_token(prev_parts)
        token = RelationToken(pos)

        return token

    @abstractmethod
    def get_attach_point(self, prev_parts):
        """
        Get the mean attachment point of where the start of the next part
        should be, given the previous ones and their relations. This function
        needs to be overridden in child classes

        :param prev_parts: TODO
        :return:
            pos: TODO
        """
        pass


class RelationIndependent(Relation):
    def __init__(self, rtype, nprev, sigma_x, sigma_y, gpos=None):
        assert rtype == 'unihist'
        Relation.__init__(self, rtype, nprev, sigma_x, sigma_y)
        self.gpos = gpos

    def get_attach_point(self, prev_parts):
        pos = self.gpos

        return pos


class RelationAttach(Relation):
    def __init__(self, rtype, nprev, sigma_x, sigma_y, attach_spot):
        assert rtype in ['start', 'end', 'mid']
        Relation.__init__(self, rtype, nprev, sigma_x, sigma_y)
        self.attach_spot = attach_spot

    def get_attach_point(self, prev_parts):
        part = prev_parts[self.attach_spot]
        if self.type == 'start':
            subtraj = part.motor[0]
            pos = subtraj[0]
        else:
            assert self.type == 'end'
            subtraj = part.motor[-1]
            pos = subtraj[-1]

        return pos


class RelationAttachAlong(RelationAttach):
    def __init__(
            self, rtype, nprev, sigma_x, sigma_y, sigma_attach, attach_spot,
            subid_spot, ncpt, eval_spot_type
    ):
        assert rtype == 'mid'
        RelationAttach.__init__(
            self, rtype, nprev, sigma_x, sigma_y, attach_spot
        )
        self.subid_spot = subid_spot
        self.ncpt = ncpt
        self.eval_spot_dist = dist.normal.Normal(eval_spot_type, sigma_attach)

    def get_attach_point(self, prev_parts):
        eval_spot_token = self.sample_eval_spot_token()
        part = prev_parts[self.attach_spot]
        bspline = part.motor_spline[:, :, self.subid_spot]
        pos = bspline_eval[eval_spot_token, bspline]

        return pos

    def sample_eval_spot_token(self):
        ll = torch.tensor(-np.inf)
        while np.isinf(ll):
            eval_spot_token = self.eval_spot_dist.sample()
            score = self.score_eval_spot_token(eval_spot_token)

        return eval_spot_token

    def score_eval_spot_token(self, eval_spot_token):
        assert type(eval_spot_token) in [int, float] or \
               (type(eval_spot_token) == torch.Tensor and
                eval_spot_token.shape == torch.Size([]))
        _, lb, ub = bspline_gen_s(self.ncpt, 1)
        if eval_spot_token < lb or eval_spot_token > ub:
            ll = torch.tensor(-np.inf)
            return ll
        ll = self.eval_spot_dist.log_prob(eval_spot_token)

        # correction for bounds
        p_within = self.eval_spot_dist.cdf(ub) - self.eval_spot_dist.cdf(lb)
        ll = ll - torch.log(p_within)

        return ll
