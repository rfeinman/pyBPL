"""
Relation class definitions
"""
from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import torch
import torch.distributions as dist

from .part import RenderedPart
from ..splines import bspline_eval, bspline_gen_s

types_allowed = ['unihist', 'start', 'end', 'mid']


class Relation(object):
    __metaclass__ = ABCMeta

    def __init__(self, rtype, sigma_x, sigma_y):
        # make sure type is valid
        assert rtype in types_allowed
        self.type = rtype

        # build the position distribution
        Cov = torch.eye(2)
        Cov[0,0] = sigma_x
        Cov[1,1] = sigma_y
        self.pos_dist = dist.multivariate_normal.MultivariateNormal(
            torch.zeros(2), Cov
        )

    def sample_position(self, prev_rendered_parts):
        for rp in prev_rendered_parts:
            assert isinstance(rp, RenderedPart)
        base = self.get_attach_point(prev_rendered_parts)
        assert base.shape == torch.Size([2])
        pos = base + self.pos_dist.sample()

        return pos

    @abstractmethod
    def get_attach_point(self, prev_rendered_parts):
        """
        Get the mean attachment point of where the start of the next part
        should be, given the previous parts (rendered). This function
        needs to be overridden in child classes.

        :param prev_parts: TODO
        :return:
            pos: TODO
        """
        pass


class RelationIndependent(Relation):
    def __init__(self, rtype, sigma_x, sigma_y, gpos):
        assert rtype == 'unihist'
        assert gpos.shape == torch.Size([2])
        super(RelationIndependent, self).__init__(rtype, sigma_x, sigma_y)
        self.gpos = gpos

    def get_attach_point(self, prev_rendered_parts):
        pos = self.gpos

        return pos


class RelationAttach(Relation):
    def __init__(self, rtype, sigma_x, sigma_y, attach_spot):
        assert rtype in ['start', 'end', 'mid']
        super(RelationAttach, self).__init__(rtype, sigma_x, sigma_y)
        self.attach_spot = attach_spot

    def get_attach_point(self, prev_rendered_parts):
        # TODO - This should be generalized so that it is applicable to all
        # TODO - types of relations. Right now motor/motor_spline is specific
        # TODO - to characters.
        rendered_part = prev_rendered_parts[self.attach_spot]
        if self.type == 'start':
            subtraj = rendered_part.motor[0]
            pos = subtraj[0]
        else:
            assert self.type == 'end'
            subtraj = rendered_part.motor[-1]
            pos = subtraj[-1]

        return pos


class RelationAttachAlong(RelationAttach):
    def __init__(
            self, rtype, sigma_x, sigma_y, sigma_attach, attach_spot,
            subid_spot, ncpt, eval_spot_type
    ):
        assert rtype == 'mid'
        super(RelationAttachAlong, self).__init__(
            rtype, sigma_x, sigma_y, attach_spot
        )
        self.subid_spot = subid_spot
        self.ncpt = ncpt
        self.eval_spot_dist = dist.normal.Normal(eval_spot_type, sigma_attach)

    def get_attach_point(self, prev_rendered_parts):
        eval_spot_token = self.sample_eval_spot_token()
        rendered_part = prev_rendered_parts[self.attach_spot]
        bspline = rendered_part.motor_spline[:, :, self.subid_spot]
        pos, _ = bspline_eval(eval_spot_token, bspline)
        # convert (1,2) tensor -> (2,) tensor
        pos = torch.squeeze(pos, dim=0)

        return pos

    def sample_eval_spot_token(self):
        ll = torch.tensor(-float('inf'))
        while ll == -float('inf'):
            eval_spot_token = self.eval_spot_dist.sample()
            ll = self.score_eval_spot_token(eval_spot_token)

        return eval_spot_token

    def score_eval_spot_token(self, eval_spot_token):
        assert type(eval_spot_token) in [int, float] or \
               (type(eval_spot_token) == torch.Tensor and
                eval_spot_token.shape == torch.Size([]))
        _, lb, ub = bspline_gen_s(self.ncpt, 1)
        if eval_spot_token < lb or eval_spot_token > ub:
            ll = torch.tensor(-float('inf'))
            return ll
        ll = self.eval_spot_dist.log_prob(eval_spot_token)

        # correction for bounds
        p_within = self.eval_spot_dist.cdf(ub) - self.eval_spot_dist.cdf(lb)
        ll = ll - torch.log(p_within)

        return ll
