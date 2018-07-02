from __future__ import division, print_function
import torch
import torch.distributions as dist

from ..concept.ctd import ConceptTypeDist
from ..concept.relation import (RelationIndependent, RelationAttach,
                                RelationAttachAlong)
from ..character.stroke import Stroke
from ..splines import bspline_gen_s
from .. import CPDUnif


class CharacterTypeDist(ConceptTypeDist):
    relation_types = ['unihist', 'start', 'end', 'mid']

    def __init__(self, lib):
        super(CharacterTypeDist, self).__init__()
        assert len(lib.pkappa.shape) == 1
        # number of control points
        self.ncpt = lib.ncpt
        # distribution of 'k' (number of strokes)
        self.kappa = dist.Categorical(probs=lib.pkappa)
        # distribution of unihist relation positions
        self.Spatial = lib.Spatial
        # distribution of relation types
        self.rel_mixdist = dist.Categorical(probs=lib.rel['mixprob'])
        # token-level variance distributions for relations
        mu = torch.zeros(2)
        Cov = torch.tensor([[lib.rel['sigma_x'],0.], [0.,lib.rel['sigma_y']]])
        self.rel_pos_dist = dist.MultivariateNormal(mu, Cov)
        self.rel_sigma_attach = lib.tokenvar['sigma_attach']
        # substroke distributions
        self.pmat_nsub = lib.pmat_nsub
        self.logStart = lib.logStart
        self.pT = lib.pT
        # shapes distribution
        if lib.isunif:
            self.sample_shapes_type = \
                lambda subid: CPDUnif.sample_shape_type(lib, subid)
        else:
            shape_mu = lib.shape['mu']
            shape_Cov = lib.shape['Sigma'].permute([2,0,1])
            self.sample_shapes_type = \
                lambda subid: sample_shapes_type(shape_mu, shape_Cov, subid, self.ncpt)

    def sample_k(self):
        """
        See ConceptTypeDist.sample_k for description
        """
        # sample from kappa
        # NOTE: add 1 to 0-indexed samples
        k = self.kappa.sample() + 1

        return k

    def score_k(self, k):
        """
        See ConceptTypeDist.score_k for description
        """
        # check if any values are out of bounds
        if k > len(self.kappa.probs):
            ll = torch.tensor(-float('Inf'))
        else:
            # score points using kappa
            # NOTE: subtract 1 to get 0-indexed samples
            ll = self.kappa.log_prob(k-1)

        return ll

    def sample_part_type(self, k):
        """
        See ConceptTypeDist.sample_part_type for description
        """
        # sample the number of sub-strokes
        nsub = sample_nsub(self.pmat_nsub, k)
        # sample the sub-stroke sequence
        ss_seq = sample_sequence(self.logStart, self.pT, nsub)
        # sample control points for each sub-stroke in the sequence
        cpts = self.sample_shapes_type(ss_seq)
        # sample scales for each sub-stroke in the sequence
        scales = CPD.sample_invscale_type(lib, ss_seq)
        # initialize the stroke type
        stroke = Stroke(
            ss_seq, cpts, scales,
            sigma_shape=lib.tokenvar['sigma_shape'],
            sigma_invscale=lib.tokenvar['sigma_invscale']
        )

        return stroke

    def sample_relation_type(self, prev_parts):
        """
        See ConceptTypeDist.sample_relation_type for description
        """
        nprev = len(prev_parts)
        stroke_num = nprev + 1
        ncpt = self.ncpt
        pos_dist = self.rel_pos_dist
        sigma_attach = self.rel_sigma_attach
        if nprev == 0:
            rtype = 'unihist'
        else:
            indx = self.rel_mixdist.sample()
            rtype = self.relation_types[indx]

        if rtype == 'unihist':
            data_id = torch.tensor([stroke_num])
            gpos = self.Spatial.sample(data_id)
            # convert (1,2) tensor to (2,) tensor
            gpos = torch.squeeze(gpos)
            # create relation
            r = RelationIndependent(rtype, pos_dist, gpos)
        elif rtype in ['start', 'end']:
            # sample random attach spot uniformly
            probs = torch.ones(nprev, requires_grad=True)
            attach_spot = dist.Categorical(probs=probs).sample()
            # create relation
            r = RelationAttach(rtype, pos_dist, attach_spot)
        elif rtype == 'mid':
            # sample random attach spot uniformly
            probs = torch.ones(nprev, requires_grad=True)
            attach_spot = dist.Categorical(probs=probs).sample()
            # sample random subid spot uniformly
            nsub = prev_parts[attach_spot].nsub
            probs = torch.ones(nsub, requires_grad=True)
            subid_spot = dist.Categorical(probs=probs).sample()
            # sample eval_spot_type
            _, lb, ub = bspline_gen_s(ncpt, 1)
            eval_spot_type = dist.Uniform(lb, ub).sample()
            # create relation
            r = RelationAttachAlong(
                rtype, pos_dist, sigma_attach, attach_spot,
                subid_spot, ncpt, eval_spot_type
            )
        else:
            raise TypeError('invalid relation')

        return r

# ----
# Substrokes model helper functions
# ----

def sample_nsub(pmat_nsub, k, nsamp=1):
    """
    Sample a sub-stroke count (or a vector of sub-stroke counts if nsamp>1)

    :param pmat_nsub: TODO
    :param k: [tensor] stroke count. scalar
    :param nsamp: [int] number of samples to draw
    :return:
        nsub: [(n,) tensor] vector of sub-stroke counts. scalar if nsamp=1
    """
    # probability of each sub-stroke count, conditioned on the number of strokes
    # NOTE: subtract 1 from stroke counts to get Python index
    pvec = pmat_nsub[k-1]
    # make sure pvec is a vector
    assert len(pvec.shape) == 1
    # sample from the categorical distribution. Add 1 to 0-indexed samples
    nsub = dist.Categorical(probs=pvec).sample(torch.Size([nsamp])) + 1
    # convert vector to scalar if nsamp=1
    nsub = torch.squeeze(nsub)

    return nsub

def sample_sequence(logStart, pT_func, nsub, nsamp=1):
    """
    Sample the sequence of sub-strokes for this stroke

    :param logStart: TODO
    :param pT_func: TODO
    :param nsub: [tensor] scalar; sub-stroke count
    :param nsamp: [int] number of samples to draw
    :return:
        samps: [(nsamp, nsub) tensor] matrix of sequence samples. vector if
                nsamp=1
    """
    # nsub should be a scalar
    assert nsub.shape == torch.Size([])

    samps = []
    for _ in range(nsamp):
        # set initial transition probabilities
        pT = torch.exp(logStart)
        # sub-stroke sequence is a list
        seq = []
        # step through and sample 'nsub' sub-strokes
        for _ in range(nsub):
            # sample the sub-stroke
            ss = dist.Categorical(probs=pT).sample()
            seq.append(ss)
            # update transition probabilities; condition on previous sub-stroke
            pT = pT_func(ss)
        # convert list into tensor
        seq = torch.tensor(seq)
        samps.append(seq.view(1,-1))
    # concatenate list of samples into tensor (matrix)
    samps = torch.cat(samps)
    # if nsamp=1 this should be a vector
    samps = torch.squeeze(samps, dim=0)

    return samps

# ----
# Shapes model helper functions
# ----

def get_shapes_dist(mu, Cov, subid):
    """
    TODO

    :param mu:
    :param Cov:
    :param subid:
    :return:
    """
    assert len(mu.shape) == 2
    assert len(Cov.shape) == 3
    assert mu.shape[0] == Cov.shape[0]
    assert Cov.shape[1] == Cov.shape[2]
    # get sub-set of mu and Cov according to subid
    Cov_sub = Cov[subid]
    mu_sub = mu[subid]
    mvn = dist.MultivariateNormal(mu_sub, Cov_sub)

    return mvn

def sample_shapes_type(mu, Cov, subid, ncpt):
    """
    Sample the control points for each sub-stroke

    :param mu: TODO
    :param Cov: TODO
    :param subid: [(nsub,) tensor] vector of sub-stroke ids
    :param ncpt: TODO
    :return:
        bspline_stack: [(ncpt, 2, nsub) tensor] sampled spline
    """
    # check that it is a vector
    assert len(subid.shape) == 1
    # record vector length
    nsub = len(subid)
    # create multivariate normal distribution
    mvn = get_shapes_dist(mu, Cov, subid)
    # sample points from the multivariate normal distribution
    rows_bspline = mvn.sample()
    # convert (nsub, ncpt*2) tensor into (ncpt, 2, nsub) tensor
    bspline_stack = torch.transpose(rows_bspline,0,1).view(ncpt,2,nsub)

    return bspline_stack