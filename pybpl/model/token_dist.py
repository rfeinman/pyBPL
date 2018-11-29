from __future__ import division, print_function
from abc import ABCMeta, abstractmethod
import torch
import torch.distributions as dist

from ..parameters import defaultps
from ..splines import bspline_gen_s
from ..part import PartType, StrokeType, PartToken, StrokeToken
from ..relation import RelationToken
from ..concept import ConceptType, CharacterType, ConceptToken, CharacterToken


class ConceptTokenDist(object):
    """
    Defines the distribution P(Token | Type) for concepts
    """
    __metaclass__ = ABCMeta

    def __init__(self, lib):
        self.pdist = PartTokenDist(lib)
        self.rdist = RelationTokenDist(lib)

    @abstractmethod
    def sample_location(self, rtoken, prev_parts):
        pass

    @abstractmethod
    def score_location(self, rtoken, prev_parts, loc):
        pass

    @abstractmethod
    def sample_token(self, ctype):
        """
        Parameters
        ----------
        ctype : ConceptType

        Returns
        -------
        ctoken : ConceptToken
        """
        assert isinstance(ctype, ConceptType)
        P = []
        R = []
        for p, r in zip(ctype.part_types, ctype.relation_types):
            # sample part token
            ptoken = self.pdist.sample_part_token(p)
            # sample relation token
            rtoken = self.rdist.sample_relation_token(r)
            # sample part position from relation token
            ptoken.position = self.sample_location(rtoken, P)
            # append them to the list
            P.append(ptoken)
            R.append(rtoken)
        ctoken = ConceptToken(P, R)

        return ctoken

    @abstractmethod
    def score_token(self, ctype, ctoken):
        """
        Parameters
        ----------
        ctype : ConceptType
        ctoken : ConceptToken

        Returns
        -------
        ll : tensor
        """
        ll = 0.
        for i in range(ctype.k):
            ll = ll + self.pdist.score_part_token(
                ctype.part_types[i], ctoken.part_tokens[i]
            )
            ll = ll + self.rdist.score_relation_token(
                ctype.relation_types[i], ctoken.relation_tokens[i]
            )
            ll = ll + self.score_location(
                ctoken.relation_tokens[i], ctoken.part_tokens[:i],
                ctoken.part_tokens[i].position
            )

        return ll


class CharacterTokenDist(ConceptTokenDist):
    """
    Defines the distribution P(Token | Type) for characters
    """
    def __init__(self, lib):
        super(CharacterTokenDist, self).__init__(lib)
        self.pdist = StrokeTokenDist(lib)
        self.default_ps = defaultps()
        # token-level position distribution parameters
        sigma_x = lib.rel['sigma_x']
        sigma_y = lib.rel['sigma_y']
        loc_Cov = torch.diag(torch.stack([sigma_x, sigma_y]))
        self.loc_dist = dist.MultivariateNormal(torch.zeros(2), loc_Cov)

    def sample_location(self, rtoken, prev_parts):
        """
        Sample a location for a given part

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
        base = rtoken.get_attach_point(prev_parts)
        assert base.shape == torch.Size([2])
        loc = base + self.loc_dist.sample()

        return loc

    def score_location(self, rtoken, prev_parts, loc):
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
        base = rtoken.get_attach_point(prev_parts)
        assert base.shape == torch.Size([2])
        ll = self.loc_dist.log_prob(loc - base)

        return ll

    def sample_affine(self):
        """
        Sample an affine warp
        TODO: update this function. right now it returns None

        Returns
        -------
        affine : (4,) tensor
            affine transformation
        """
        # set affine to None for now
        affine = None

        return affine

    def score_affine(self, affine):
        return 0.

    def sample_image_noise(self):
        """
        Sample an "epsilon," i.e. image noise quantity
        TODO: update this function. right now it returns fixed quantity

        Returns
        -------
        epsilon : tensor
            scalar; image noise quantity
        """
        # set rendering parameters to minimum noise for now
        epsilon = self.default_ps.min_epsilon

        return epsilon

    def score_image_noise(self, epsilon):
        return 0.

    def sample_image_blur(self):
        """
        Sample a "blur_sigma," i.e. image blur quantity.

        Returns
        -------
        blur_sigma: tensor
            scalar; image blur quantity
        """
        # set rendering parameters to minimum noise for now
        lb = self.default_ps.min_blur_sigma
        ub = self.default_ps.max_blur_sigma
        blur_sigma = dist.Uniform(lb, ub).sample()

        return blur_sigma

    def score_image_blur(self, blur_sigma):
        """
        Compute the log-probability of an image blur quantity.

        Parameters
        ----------
        blur_sigma : tensor
            scalar; image blur quantity

        Returns
        -------
        ll : tensor
            scalar; log-probability of the image blur

        """
        lb = self.default_ps.min_blur_sigma
        ub = self.default_ps.max_blur_sigma
        ll = dist.Uniform(lb, ub).log_prob(blur_sigma)

        return ll

    def sample_token(self, ctype):
        """
        Sample a character token from P(Token | Type = type).
        Note: should only be called from Model

        Parameters
        ----------
        ctype : CharacterType
            TODO

        Returns
        -------
        ctoken : CharacterToken
            character token
        """
        # sample part and relation tokens
        concept_token = super(CharacterTokenDist, self).sample_token(ctype)

        # sample affine warp
        affine = self.sample_affine() # (4,) tensor

        # sample rendering parameters
        epsilon = self.sample_image_noise()
        blur_sigma = self.sample_image_blur()

        # create the character token
        ctoken = CharacterToken(
            concept_token.part_tokens, concept_token.relation_tokens, affine,
            epsilon, blur_sigma
        )

        return ctoken

    def score_token(self, ctype, ctoken):
        """
        Compute the log-probability of a concept token,
        log P(Token = token | Type = type).
        Note: Should only be called from Model

        Parameters
        ----------
        ctype : CharacterType
            concept type to condition on
        ctoken : CharacterToken
            concept token to be scored

        Returns
        -------
        ll : tensor
            scalar; log-likelihood of the token
        """
        ll = super(CharacterTokenDist, self).score_token(ctype, ctoken)
        ll += self.score_affine(ctoken.affine)
        ll += self.score_image_noise(ctoken.epsilon)
        ll += self.score_image_blur(ctoken.blur_sigma)

        return ll


class PartTokenDist(object):
    __metaclass__ = ABCMeta
    def __init__(self, lib):
        pass

    @abstractmethod
    def sample_part_token(self, ptype):
        pass

    @abstractmethod
    def score_part_token(self, ptype, ptoken):
        pass


class StrokeTokenDist(PartTokenDist):
    def __init__(self, lib):
        super(StrokeTokenDist, self).__init__(lib)
        # shapes distribution params
        self.sigma_shape = lib.tokenvar['sigma_shape']
        # invscale distribution params
        self.sigma_invscale = lib.tokenvar['sigma_invscale']
        self.xlim = lib.Spatial.xlim
        self.ylim = lib.Spatial.ylim

    def sample_shapes_token(self, shapes_type):
        """
        Sample a token of each sub-stroke's shapes

        Parameters
        ----------
        shapes_type : (ncpt, 2, nsub) tensor
            shapes type to condition on

        Returns
        -------
        shapes_token : (ncpt, 2, nsub) tensor
            sampled shapes token
        """
        shapes_dist = dist.normal.Normal(shapes_type, self.sigma_shape)
        # sample shapes token
        shapes_token = shapes_dist.sample()

        return shapes_token

    def score_shapes_token(self, shapes_type, shapes_token):
        """
        Compute the log-likelihood of each sub-strokes's shapes

        Parameters
        ----------
        shapes_type : (ncpt, 2, nsub) tensor
            shapes type to condition on
        shapes_token : (ncpt, 2, nsub) tensor
            shapes tokens to score

        Returns
        -------
        ll : (nsub,) tensor
            vector of log-likelihood scores
        """
        shapes_dist = dist.normal.Normal(shapes_type, self.sigma_shape)
        # compute scores for every element in shapes_token
        ll = shapes_dist.log_prob(shapes_token)

        return ll

    def sample_invscales_token(self, invscales_type):
        """
        Sample a token of each sub-stroke's scale

        Parameters
        ----------
        invscales_type : (nsub,) tensor
            invscales type to condition on

        Returns
        -------
        invscales_token : (nsub,) tensor
            sampled invscales token
        """
        scales_dist = dist.normal.Normal(invscales_type, self.sigma_invscale)
        while True:
            invscales_token = scales_dist.sample()
            ll = self.score_invscales_token(invscales_type, invscales_token)
            if not torch.any(ll == -float('inf')):
                break

        return invscales_token

    def score_invscales_token(self, invscales_type, invscales_token):
        """
        Compute the log-likelihood of each sub-stroke's scale

        Parameters
        ----------
        invscales_type : (nsub,) tensor
            invscales type to condition on
        invscales_token : (nsub,) tensor
            scales tokens to score

        Returns
        -------
        ll : (nsub,) tensor
            vector of log-likelihood scores
        """
        scales_dist = dist.normal.Normal(invscales_type, self.sigma_invscale)
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

    def sample_part_token(self, ptype):
        """
        Sample a stroke token

        Parameters
        ----------
        ptype : StrokeType
            stroke type to condition on

        Returns
        -------
        ptoken : StrokeToken
            stroke token sample
        """
        shapes_token = self.sample_shapes_token(ptype.shapes)
        invscales_token = self.sample_invscales_token(ptype.invscales)
        ptoken = StrokeToken(
            shapes_token, invscales_token, self.xlim, self.ylim
        )

        return ptoken

    def score_part_token(self, ptype, ptoken):
        """
        Compute the log-likelihood of a stroke token

        Parameters
        ----------
        ptype : StrokeType
            stroke type to condition on
        ptoken : StrokeToken
            stroke token to score

        Returns
        -------
        ll : tensor
            scalar; log-likelihood of the stroke token
        """
        shapes_scores = self.score_shapes_token(ptype.shapes, ptoken.shapes)
        invscales_scores = self.score_invscales_token(
            ptype.invscales, ptoken.invscales
        )
        ll = torch.sum(shapes_scores) + torch.sum(invscales_scores)

        return ll


class RelationTokenDist(object):

    def __init__(self, lib):
        # number of control points
        self.ncpt = lib.ncpt
        # eval_spot_token distribution params
        self.sigma_attach = lib.tokenvar['sigma_attach']

    def sample_relation_token(self, rtype):
        """
        Parameters
        ----------
        rtype : Relation

        Returns
        -------
        rtoken : RelationToken
        """
        if rtype.category == 'mid':
            assert hasattr(rtype, 'eval_spot')
            eval_spot_dist = dist.normal.Normal(
                rtype.eval_spot, self.sigma_attach
            )
            eval_spot_token = sample_eval_spot_token(
                eval_spot_dist, self.ncpt
            )
            rtoken = RelationToken(rtype, eval_spot_token=eval_spot_token)
        else:
            rtoken = RelationToken(rtype)

        return rtoken

    def score_relation_token(self, rtype, rtoken):
        """

        Parameters
        ----------
        rtype : Relation
        rtoken : RelationToken

        Returns
        -------
        ll : tensor
            scalar; log-probability of relation token
        """
        if rtype.category == 'mid':
            assert hasattr(rtype, 'eval_spot')
            assert hasattr(rtoken, 'eval_spot_token')
            eval_spot_dist = dist.normal.Normal(
                rtype.eval_spot, self.sigma_attach
            )
            ll = score_eval_spot_token(
                rtoken.eval_spot_token, eval_spot_dist, self.ncpt
            )
        else:
            ll = 0.

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
    while True:
        eval_spot_token = eval_spot_dist.sample()
        ll = score_eval_spot_token(eval_spot_token, eval_spot_dist, ncpt)
        if not ll == -float('inf'):
            break

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
