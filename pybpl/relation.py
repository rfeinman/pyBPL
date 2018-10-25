"""
Relations for sampling part positions. Relations, together with parts, make up
concepts.
"""
from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import torch
import torch.distributions as dist

from .part import PartToken
from .splines import bspline_eval, bspline_gen_s

types_allowed = ['unihist', 'start', 'end', 'mid']



class Relation(object):
    """
    TODO

    Parameters
    ----------
    category : TODO
        TODO
    pos_dist : TODO
        TODO
    """
    __metaclass__ = ABCMeta

    def __init__(self, category, pos_dist):
        # make sure type is valid
        assert category in types_allowed
        self.category = category
        self.pos_dist = pos_dist

    def sample_position(self, prev_parts):
        """
        TODO

        Parameters
        ----------
        prev_parts : list of PartToken
            previous part tokens

        Returns
        -------
        pos : (2,) tensor
            position; x-y coordinates
        """
        for pt in prev_parts:
            assert isinstance(pt, PartToken)
        base = self.get_attach_point(prev_parts)
        assert base.shape == torch.Size([2])
        pos = base + self.pos_dist.sample()

        return pos

    @abstractmethod
    def get_attach_point(self, prev_parts):
        pass


class RelationIndependent(Relation):
    """
    TODO

    Parameters
    ----------
    category : TODO
        TODO
    pos_dist : TODO
        TODO
    gpos : (2,) tensor
        position; x-y coordinates
    """
    def __init__(self, category, pos_dist, gpos):
        assert category == 'unihist'
        assert gpos.shape == torch.Size([2])
        super(RelationIndependent, self).__init__(category, pos_dist)
        self.gpos = gpos

    def get_attach_point(self, prev_parts):
        """
        Get the mean attachment point of where the start of the next part
        should be, given the previous part tokens. TODO

        Parameters
        ----------
        prev_parts : TODO
            TODO

        Returns
        -------
        pos: (2,) tensor
            position; x-y coordinates
        """
        pos = self.gpos

        return pos


class RelationAttach(Relation):
    """
    TODO

    Parameters
    ----------
    category : TODO
        TODO
    pos_dist : TODO
        TODO
    attach_part : TODO
        TODO
    """
    def __init__(self, category, pos_dist, attach_part):
        assert category in ['start', 'end', 'mid']
        super(RelationAttach, self).__init__(category, pos_dist)
        self.attach_part = attach_part

    def get_attach_point(self, prev_parts):
        """
        Get the mean attachment point of where the start of the next part
        should be, given the previous part tokens. TODO

        Parameters
        ----------
        prev_parts : TODO
            TODO

        Returns
        -------
        pos: TODO
            TODO
        """
        # TODO - This should be generalized so that it is applicable to all
        # TODO - types of relations. Right now motor/motor_spline is specific
        # TODO - to characters.
        part = prev_parts[self.attach_part]
        if self.category == 'start':
            subtraj = part.motor[0]
            pos = subtraj[0]
        else:
            assert self.category == 'end'
            subtraj = part.motor[-1]
            pos = subtraj[-1]

        return pos


class RelationAttachAlong(RelationAttach):
    """
    TODO

    Parameters
    ----------
    category : TODO
        TODO
    pos_dist : TODO
        TODO
    sigma_attach : TODO
        TODO
    attach_part : TODO
        TODO
    attach_subid : TODO
        TODO
    ncpt : TODO
        TODO
    eval_spot_type : TODO
        TODO
    """
    def __init__(
            self, category, pos_dist, sigma_attach, attach_part,
            attach_subid, ncpt, eval_spot_type
    ):
        assert category == 'mid'
        super(RelationAttachAlong, self).__init__(category, pos_dist, attach_part)
        self.attach_subid = attach_subid
        self.ncpt = ncpt
        self.eval_spot_dist = dist.normal.Normal(eval_spot_type, sigma_attach)

    def get_attach_point(self, prev_parts):
        """
        Get the mean attachment point of where the start of the next part
        should be, given the previous part tokens. TODO

        Parameters
        ----------
        prev_parts : TODO
            TODO

        Returns
        -------
        pos : TODO
            TODO
        """
        eval_spot_token = self.sample_eval_spot_token()
        part = prev_parts[self.attach_part]
        bspline = part.motor_spline[:,:,self.attach_subid]
        pos, _ = bspline_eval(eval_spot_token, bspline)
        # convert (1,2) tensor -> (2,) tensor
        pos = torch.squeeze(pos, dim=0)

        return pos

    def sample_eval_spot_token(self):
        """
        TODO

        Returns
        -------
        eval_spot_token : TODO
            TODO
        """
        ll = torch.tensor(-float('inf'))
        while ll == -float('inf'):
            eval_spot_token = self.eval_spot_dist.sample()
            ll = self.score_eval_spot_token(eval_spot_token)

        return eval_spot_token

    def score_eval_spot_token(self, eval_spot_token):
        """
        TODO

        Parameters
        ----------
        eval_spot_token : TODO
            TODO

        Returns
        -------
        ll : TODO
            TODO
        """
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
