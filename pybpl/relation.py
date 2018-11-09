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

categories_allowed = ['unihist', 'start', 'end', 'mid']




class RelationToken(object):
    """
    RelationToken instances hold all of the token-level information for a
    relation

    Parameters
    ----------
    rel : Relation
        relation type
    eval_spot_token : tensor
        Optional parameter. Token-level evaluation spot for RelationAttachAlong
    """
    def __init__(self, rel, **kwargs):
        self.rel = rel
        if rel.category in ['unihist', 'start', 'end']:
            assert kwargs == {}
        else:
            assert set(kwargs.keys()) == {'eval_spot_token'}
            self.eval_spot_token = kwargs['eval_spot_token']

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
        if self.rel.category == 'mid':
            _, lb, ub = bspline_gen_s(self.rel.ncpt, 1)
            params = [self.eval_spot_token]
            lbs = [lb]
            ubs = [ub]
        else:
            params = []
            lbs = []
            ubs = []

        return params, lbs, ubs

    def sample_location(self, prev_parts):
        """
        Sample a location from the relation token

        Parameters
        ----------
        prev_parts : list of PartToken
            previous part tokens

        Returns
        -------
        loc : (2,) tensor
            location; x-y coordinates

        """
        for pt in prev_parts:
            assert isinstance(pt, PartToken)
        base = self.get_attach_point(prev_parts)
        assert base.shape == torch.Size([2])
        loc = base + self.rel.loc_dist.sample()

        return loc

    def score_location(self, loc, prev_parts):
        """
        Compute the log-likelihood of a location

        Parameters
        ----------
        loc : (2,) tensor
            location; x-y coordinates
        prev_parts : list of PartToken
            previous part tokens

        Returns
        -------
        ll : tensor
            scalar; log-likelihood of the location

        """
        for pt in prev_parts:
            assert isinstance(pt, PartToken)
        base = self.get_attach_point(prev_parts)
        assert base.shape == torch.Size([2])
        ll = self.rel.loc_dist.log_prob(loc - base)

        return ll

    def get_attach_point(self, prev_parts):
        """
        Get the mean attachment point of where the start of the next part
        should be, given the previous part tokens.

        Parameters
        ----------
        prev_parts : list of PartToken
            previous part tokens

        Returns
        -------
        loc : (2,) tensor
            attach point (location); x-y coordinates

        """
        if self.rel.category == 'unihist':
            loc = self.rel.gpos
        else:
            prev = prev_parts[self.rel.attach_ix]
            if self.rel.category == 'start':
                subtraj = prev.motor[0]
                loc = subtraj[0]
            elif self.rel.category == 'end':
                subtraj = prev.motor[-1]
                loc = subtraj[-1]
            else:
                assert self.rel.category == 'mid'
                bspline = prev.motor_spline[:, :, self.rel.attach_subix]
                loc, _ = bspline_eval(self.eval_spot_token, bspline)
                # convert (1,2) tensor -> (2,) tensor
                loc = torch.squeeze(loc, dim=0)

        return loc


class Relation(object):
    """
    Relations define the relationship between the current part and all previous
    parts. They fall into 4 categories: ['unihist','start','end','mid']. Holds
    all type-level parameters of the relation

    Parameters
    ----------
    category : string
        relation category
    lib : Library
        library instance, which holds token-level distribution parameters
    """
    __metaclass__ = ABCMeta

    def __init__(self, category, lib):
        # make sure type is valid
        assert category in categories_allowed
        self.category = category
        # token-level position distribution parameters
        sigma_x = lib.rel['sigma_x']
        sigma_y = lib.rel['sigma_y']
        loc_Cov = torch.tensor([[sigma_x, 0.], [0., sigma_y]])
        self.loc_dist = dist.MultivariateNormal(torch.zeros(2), loc_Cov)

    @abstractmethod
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
        pass

    def sample_token(self):
        """
        Sample a token of the relation

        Returns
        -------
        token : RelationToken
            relation token sample
        """
        token = RelationToken(self)

        return token

    def score_token(self, token):
        """
        Compute the log-likelihood of a relation token

        Parameters
        ----------
        token : RelationToken
            relation token to score

        Returns
        -------
        ll : tensor
            scalar; log-likelihood of the relation token

        """
        # default this to 0. The only relation category that has token-level
        # parameters is the 'mid'. For 'mid' this function is over-ridden
        # (see RelationAttachAlong)
        ll = torch.tensor(0.)

        return ll


class RelationIndependent(Relation):
    """
    TODO

    Parameters
    ----------
    category : string
        relation category
    gpos : (2,) tensor
        position; x-y coordinates
    xlim : (2,) tensor
        [lower, upper]; bounds for the x direction
    ylim : (2,) tensor
        [lower, upper]; bounds for the y direction
    lib : Library
        library instance, which holds token-level distribution parameters
    """
    def __init__(self, category, gpos, xlim, ylim, lib):
        super(RelationIndependent, self).__init__(category, lib)
        assert category == 'unihist'
        assert gpos.shape == torch.Size([2])
        self.gpos = gpos
        self.xlim = xlim
        self.ylim = ylim

    def optimizable_parameters(self, eps=1e-4):
        params = [self.gpos]
        bounds = torch.cat([self.xlim.view(1, -1), self.ylim.view(1, -1)])
        lbs = [bounds[:,0]]
        ubs = [bounds[:,1]]

        return params, lbs, ubs


class RelationAttach(Relation):
    """
    TODO

    Parameters
    ----------
    category : string
        relation category
    attach_ix : int
        index of previous part to which this part will attach
    lib : Library
        library instance, which holds token-level distribution parameters
    """
    def __init__(self, category, attach_ix, lib):
        super(RelationAttach, self).__init__(category, lib)
        assert category in ['start', 'end', 'mid']
        self.attach_ix = attach_ix

    def optimizable_parameters(self, eps=1e-4):
        params = []
        lbs = []
        ubs = []

        return params, lbs, ubs


class RelationAttachAlong(RelationAttach):
    """
    TODO

    Parameters
    ----------
    category : string
        relation category
    attach_ix : int
        index of previous part to which this part will attach
    attach_subix : int
        index of sub-stroke from the selected previous part to which
        this part will attach
    eval_spot : tensor
        type-level spline coordinate
    lib : Library
        library instance, which holds token-level distribution parameters
    """
    def __init__(self, category, attach_ix, attach_subix, eval_spot, lib):
        super(RelationAttachAlong, self).__init__(category, attach_ix, lib)
        assert category == 'mid'
        self.attach_subix = attach_subix
        self.eval_spot = eval_spot
        # token-level eval_spot distribution parameters
        self.ncpt = lib.ncpt
        self.sigma_attach = lib.tokenvar['sigma_attach']

    def optimizable_parameters(self, eps=1e-4):
        _, lb, ub = bspline_gen_s(self.ncpt, 1)
        params = [self.eval_spot]
        lbs = [lb]
        ubs = [ub]

        return params, lbs, ubs

    def sample_token(self):
        """
        Sample a token of the relation

        Returns
        -------
        token : RelationToken
            sampled relation token
        """
        eval_spot_dist = dist.normal.Normal(self.eval_spot, self.sigma_attach)
        eval_spot_token = sample_eval_spot_token(eval_spot_dist, self.ncpt)
        token = RelationToken(self, eval_spot_token=eval_spot_token)

        return token

    def score_token(self, token):
        """
        Compute the log-likelihood of a relation token

        Parameters
        ----------
        token : RelationToken
            relation token to score

        Returns
        -------
        ll : tensor
            scalar; log-likelihood of the relation token
        """
        assert hasattr(token, 'eval_spot_token')
        eval_spot_dist = dist.normal.Normal(self.eval_spot, self.sigma_attach)
        ll = score_eval_spot_token(
            token.eval_spot_token, eval_spot_dist, self.ncpt
        )

        return ll



def sample_eval_spot_token(eval_spot_dist, ncpt):
    """
    Sample an evaluation spot token

    Parameters
    ----------
    eval_spot_dist : Distribution
        torch distribution; will be used to sample evaluation spot tokens
    ncpt : int
        number of control points

    Returns
    -------
    eval_spot_token : tensor
        scalar; token-level spline coordinate
    """
    ll = torch.tensor(-float('inf'))
    while ll == -float('inf'):
        eval_spot_token = eval_spot_dist.sample()
        ll = score_eval_spot_token(eval_spot_token, eval_spot_dist, ncpt)

    return eval_spot_token


def score_eval_spot_token(eval_spot_token, eval_spot_dist, ncpt):
    """
    Compute the log-likelihood of an evaluation spot token

    Parameters
    ----------
    eval_spot_token : tensor
        scalar; token-level spline coordinate
    eval_spot_dist : Distribution
        torch distribution; will be used to score evaluation spot tokens
    ncpt : int
        number of control points

    Returns
    -------
    ll : tensor
        scalar; log-likelihood of the evaluation spot token
    """
    assert type(eval_spot_token) in [int, float] or \
           (type(eval_spot_token) == torch.Tensor and
            eval_spot_token.shape == torch.Size([]))
    _, lb, ub = bspline_gen_s(ncpt, 1)
    if eval_spot_token < lb or eval_spot_token > ub:
        ll = torch.tensor(-float('inf'), dtype=torch.float)
    else:
        ll = eval_spot_dist.log_prob(eval_spot_token)
        # correction for bounds
        p_within = eval_spot_dist.cdf(ub) - eval_spot_dist.cdf(lb)
        ll = ll - torch.log(p_within)

    return ll
