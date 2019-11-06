"""
Relations for sampling part positions. Relations, together with parts, make up
concepts.
"""
from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import torch

from .part import StrokeToken
from .splines import bspline_eval, bspline_gen_s

categories_allowed = ['unihist', 'start', 'end', 'mid']


class RelationToken(object):
    """
    RelationToken instances hold all of the token-level information for a
    relation

    Parameters
    ----------
    rtype : Relation
        relation type
    eval_spot_token : tensor
        Optional parameter. Token-level evaluation spot for RelationAttachAlong
    """
    def __init__(self, rtype, **kwargs):
        self.rtype = rtype
        if rtype.category in ['unihist', 'start', 'end']:
            assert kwargs == {}
        else:
            assert set(kwargs.keys()) == {'eval_spot_token'}
            self.eval_spot_token = kwargs['eval_spot_token']

    def get_attach_point(self, prev_parts):
        """
        Get the mean attachment point of where the start of the next part
        should be, given the previous part tokens.

        Parameters
        ----------
        prev_parts : list of StrokeToken
            previous part tokens

        Returns
        -------
        loc : (2,) tensor
            attach point (location); x-y coordinates

        """
        if self.rtype.category == 'unihist':
            loc = self.rtype.gpos
        else:
            prev = prev_parts[self.rtype.attach_ix]
            if self.rtype.category == 'start':
                subtraj = prev.motor[0]
                loc = subtraj[0]
            elif self.rtype.category == 'end':
                subtraj = prev.motor[-1]
                loc = subtraj[-1]
            else:
                assert self.rtype.category == 'mid'
                bspline = prev.motor_spline[:, :, self.rtype.attach_subix]
                loc, _ = bspline_eval(self.eval_spot_token, bspline)
                # convert (1,2) tensor -> (2,) tensor
                loc = torch.squeeze(loc, dim=0)

        return loc

    def parameters(self):
        """
        Returns a list of parameters that can be optimized via gradient descent.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        parameters : list
            optimizable parameters
        """
        if self.rtype.category == 'mid':
            parameters = [self.eval_spot_token]
        else:
            parameters = []

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
        if self.rtype.category == 'mid':
            _, lb, _ = bspline_gen_s(self.rtype.ncpt, 1)
            lbs = [lb+eps]
        else:
            lbs = []

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
        if self.rtype.category == 'mid':
            _, _, ub = bspline_gen_s(self.rtype.ncpt, 1)
            ubs = [ub-eps]
        else:
            ubs = []

        return ubs

    def train(self):
        """
        makes params require grad
        """
        for param in self.parameters():
            param.requires_grad_(True)

    def eval(self):
        """
        makes params require no grad
        """
        for param in self.parameters():
            param.requires_grad_(False)

    def to(self, device):
        """
        moves parameters to device
        TODO
        """
        pass


class RelationType(object):
    """
    Relations define the relationship between the current part and all previous
    parts. They fall into 4 categories: ['unihist','start','end','mid'].
    RelationType holds all type-level parameters of the relation.
    his is an abstract base class that must be inherited from to build specific
    categories of relations.

    Parameters
    ----------
    category : string
        relation category
    """
    __metaclass__ = ABCMeta

    def __init__(self, category):
        # make sure type is valid
        assert category in categories_allowed
        self.category = category

    @abstractmethod
    def parameters(self):
        """
        Returns a list of parameters that can be optimized via gradient descent.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        parameters : list
            optimizable parameters
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    def train(self):
        """
        makes params require grad
        """
        for param in self.parameters():
            param.requires_grad_(True)

    def eval(self):
        """
        makes params require no grad
        """
        for param in self.parameters():
            param.requires_grad_(False)

    def to(self, device):
        """
        moves parameters to device
        TODO
        """
        pass


class RelationIndependent(RelationType):
    """
    RelationIndependent (or 'unihist' relations) are assigned when the part's
    location is independent of all previous parts. The global position (gpos)
    of the part is sampled at random from the prior on positions

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
    """
    def __init__(self, category, gpos, xlim, ylim):
        super(RelationIndependent, self).__init__(category)
        assert category == 'unihist'
        assert gpos.shape == torch.Size([2])
        self.gpos = gpos
        self.xlim = xlim
        self.ylim = ylim

    def parameters(self):
        parameters = [self.gpos]

        return parameters

    def lbs(self, eps=1e-4):
        bounds = torch.stack([self.xlim, self.ylim])
        lbs = [bounds[:,0]+eps]

        return lbs

    def ubs(self, eps=1e-4):
        bounds = torch.stack([self.xlim, self.ylim])
        ubs = [bounds[:,1]-eps]

        return ubs


class RelationAttach(RelationType):
    """
    RelationAttach is assigned when the part will attach to a previous part

    Parameters
    ----------
    category : string
        relation category
    attach_ix : int
        index of previous part to which this part will attach
    """
    def __init__(self, category, attach_ix):
        super(RelationAttach, self).__init__(category)
        assert category in ['start', 'end', 'mid']
        self.attach_ix = attach_ix

    def parameters(self):
        parameters = []

        return parameters

    def lbs(self, eps=1e-4):
        lbs = []

        return lbs

    def ubs(self, eps=1e-4):
        ubs = []

        return ubs


class RelationAttachAlong(RelationAttach):
    """
    RelationAttachAlong is assigned when the part will attach to a previous
    part somewhere in the middle of that part (as opposed to the start or end)

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
    ncpt : int
        number of control points
    """
    def __init__(self, category, attach_ix, attach_subix, eval_spot, ncpt):
        super(RelationAttachAlong, self).__init__(category, attach_ix)
        assert category == 'mid'
        self.attach_subix = attach_subix
        self.eval_spot = eval_spot
        self.ncpt = ncpt

    def parameters(self):
        parameters = [self.eval_spot]

        return parameters

    def lbs(self, eps=1e-4):
        _, lb, _ = bspline_gen_s(self.ncpt, 1)
        lbs = [lb+eps]

        return lbs

    def ubs(self, eps=1e-4):
        _, _, ub = bspline_gen_s(self.ncpt, 1)
        ubs = [ub-eps]

        return ubs
