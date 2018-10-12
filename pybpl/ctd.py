from __future__ import division, print_function
from abc import ABCMeta, abstractmethod
import torch
import torch.distributions as dist

from .library import Library
from .relation import RelationIndependent, RelationAttach, RelationAttachAlong
from .part import Stroke
from .splines import bspline_gen_s

# list of acceptable dtypes for 'k' parameter
int_types = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]


class ConceptTypeDist(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def sample_k(self):
        """
        Sample a number of parts from the prior

        :return:
            k: [tensor] scalar; part count
        """
        pass

    @abstractmethod
    def score_k(self, k):
        """
        Score the log-probability of the number of parts under the prior

        :param k: [tensor] scalar; part count to score
        :return:
            ll: [tensor] scalar; log-probability of the part count
        """
        pass

    @abstractmethod
    def sample_part_type(self, k):
        """
        Sample a part type from the prior, conditioned on a number of parts

        :param k: [tensor]
        :return:
            p: [Part]
        """
        pass

    @abstractmethod
    def score_part_type(self, k, p):
        """
        Score the log-probability of the part type under the prior, conditioned
        on a number of parts

        :param k: [tensor] scalar; part count
        :param p: [Part] part type to score
        :return:
            ll: [tensor] scalar; log-probability of the part type
        """
        pass

    @abstractmethod
    def sample_relation_type(self, prev_parts):
        """
        Sample a relation type from the prior for the current part, conditioned
        on the previous parts

        :param prev_parts: [list of Parts] TODO
        :return:
            r: [Relation] relation type
        """
        pass

    @abstractmethod
    def score_relation_type(self, prev_parts, r):
        """
        Score the relation type of the current part under the prior

        :param prev_parts: [list of Parts] TODO
        :param r: [Relation] TODO
        :return:
            ll: [tensor] log-probability of the relation type
        """
        pass

    def sample_type(self, k=None):
        """
        Sample a concept type from the prior

        :param k: [int or tensor] scalar; the number of parts to use
        :return:
            P: [list of Part] TODO
            R: [list of Part] TODO
        """
        if k is None:
            # sample the number of parts 'k'
            k = self.sample_k()
        elif isinstance(k, int):
            k = torch.tensor(k)
        else:
            assert isinstance(k, torch.Tensor)
            assert k.shape == torch.Size([])
            assert k.dtype in int_types

        # initialize part and relation lists
        P = []
        R = []
        # for each part, sample part parameters
        for _ in range(k):
            # sample the part type
            part = self.sample_part_type(k)
            # sample the relation type
            relation = self.sample_relation_type(P)
            # append part and relation types to their respective lists
            P.append(part)
            R.append(relation)

        # return the concept type (a stencil for a concept)
        return P, R

    def score_type(self, P, R):
        """
        Score a concept type under the prior
        P(type) = P(k)*\prod_{i=1}^k [P(S_i)P(R_i|S_{0:i-1})]

        :param P: [list of Part] TODO
        :param R: [list of Relation] TODO
        :return:
            ll: [tensor] scalar; the log-probability of the concept type
        """
        # score the number of parts
        k = len(P)
        ll = self.score_k(k)
        for i in range(k):
            ll = ll + self.score_part_type(k, P[i])
            ll = ll + self.score_relation_type(P[:i], R[i])

        return ll



class CharacterTypeDist(ConceptTypeDist):
    """
    A CharacterTypeDist is a probabilistic program that samples Character types
    from the prior. It can also score the log-likelihood of Character types.
    """
    relation_types = ['unihist', 'start', 'end', 'mid']

    def __init__(self, lib):
        """
        Initialize the CharacterTypeDist instance

        :param lib:
        """
        super(CharacterTypeDist, self).__init__()
        assert isinstance(lib, Library)
        # is uniform?
        self.isunif = lib.isunif
        # number of control points
        self.ncpt = lib.ncpt
        # distribution of 'k' (number of strokes)
        assert len(lib.pkappa.shape) == 1
        self.kappa = dist.Categorical(probs=lib.pkappa)
        # distribution of unihist relation positions
        self.Spatial = lib.Spatial
        # distribution of relation types
        self.rel_mixdist = dist.Categorical(probs=lib.rel['mixprob'])
        # token-level variance relations parameters
        pos_mu = torch.zeros(2)
        pos_Cov = torch.tensor([[lib.rel['sigma_x'],0.], [0.,lib.rel['sigma_y']]])
        self.rel_pos_dist = dist.MultivariateNormal(pos_mu, pos_Cov)
        self.rel_sigma_attach = lib.tokenvar['sigma_attach']
        # substroke distributions
        self.pmat_nsub = lib.pmat_nsub
        self.logStart = lib.logStart
        self.pT = lib.pT
        # shapes distribution
        shapes_mu = lib.shape['mu']
        shapes_Cov = lib.shape['Sigma'].permute([2,0,1])
        assert len(shapes_mu.shape) == 2
        assert len(shapes_Cov.shape) == 3
        assert shapes_mu.shape[0] == shapes_Cov.shape[0]
        assert shapes_Cov.shape[1] == shapes_Cov.shape[2]
        self.shapes_mu = shapes_mu
        self.shapes_Cov = shapes_Cov
        self.newscale = lib.newscale
        # token-level shapes parameters
        self.sigma_shape = lib.tokenvar['sigma_shape']
        self.sigma_invscale = lib.tokenvar['sigma_invscale']
        # invscales distribution
        scales_theta = lib.scale['theta']
        assert len(scales_theta.shape) == 2
        self.scales_con = scales_theta[:,0] # gamma concentration
        # NOTE: PyTorch gamma dist uses rate parameter, which is inv of scale
        self.scales_rate = 1/scales_theta[:,1] # gamma rate


    # ----
    # Num strokes model methods
    # ----

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


    # ----
    # Sub-strokes model methods
    # ----

    def sample_nsub(self, k, nsamp=1):
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
        pvec = self.pmat_nsub[k-1]
        # make sure pvec is a vector
        assert len(pvec.shape) == 1
        # sample from the categorical distribution. Add 1 to 0-indexed samples
        nsub = dist.Categorical(probs=pvec).sample(torch.Size([nsamp])) + 1
        # convert vector to scalar if nsamp=1
        nsub = torch.squeeze(nsub)

        return nsub

    def score_nsub(self, k, nsub):
        raise NotImplementedError

    def sample_subid(self, nsub, nsamp=1):
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
            pT = torch.exp(self.logStart)
            # sub-stroke sequence is a list
            seq = []
            # step through and sample 'nsub' sub-strokes
            for _ in range(nsub):
                # sample the sub-stroke
                ss = dist.Categorical(probs=pT).sample()
                seq.append(ss)
                # update transition probabilities; condition on previous sub-stroke
                pT = self.pT(ss)
            # convert list into tensor
            seq = torch.tensor(seq)
            samps.append(seq.view(1, -1))
        # concatenate list of samples into tensor (matrix)
        samps = torch.cat(samps)
        # if nsamp=1 this should be a vector
        samps = torch.squeeze(samps, dim=0)

        return samps

    def score_subid(self, nsub, subid):
        raise NotImplementedError


    # ----
    # Shapes model methods
    # ----

    def sample_shapes_type(self, subid):
        """
        Sample the control points for each sub-stroke

        :param subid: [(nsub,) tensor] vector of sub-stroke ids
        :return:
            bspline_stack: [(ncpt, 2, nsub) tensor] sampled spline
        """
        if self.isunif:
            raise NotImplementedError
        # record num control points
        ncpt = self.ncpt
        # check that it is a vector
        assert len(subid.shape) == 1
        # record vector length
        nsub = len(subid)
        # create multivariate normal distribution
        mvn = dist.MultivariateNormal(
            self.shapes_mu[subid], self.shapes_Cov[subid]
        )
        # sample points from the multivariate normal distribution
        rows_bspline = mvn.sample()
        # convert (nsub, ncpt*2) tensor into (ncpt, 2, nsub) tensor
        bspline_stack = torch.transpose(rows_bspline, 0, 1).view(ncpt, 2, nsub)

        return bspline_stack

    def score_shapes_type(self, subid, bspline_stack):
        """
        Score the log-likelihoods of the control points for each sub-stroke
        :param lib: [Library] library class instance
        :param bspline_stack: [(ncpt, 2, nsub) tensor] shapes of bsplines
        :param subid: [(nsub,) tensor] vector of sub-stroke ids
        :return:
            ll: [(nsub,) tensor] vector of log-likelihood scores
        """
        if self.isunif:
            raise NotImplementedError
        # record num control points
        ncpt = self.ncpt
        # check that it is a vector
        assert len(subid.shape) == 1
        # record vector length
        nsub = len(subid)
        assert bspline_stack.shape[-1] == nsub
        # convert (ncpt, 2, nsub) tensor into (nsub, ncpt*2) tensor
        rows_bspline = torch.transpose(bspline_stack.view(ncpt*2, nsub), 0, 1)
        # create multivariate normal distribution
        mvn = dist.MultivariateNormal(
            self.shapes_mu[subid], self.shapes_Cov[subid]
        )
        # score points using the multivariate normal distribution
        ll = mvn.log_prob(rows_bspline)

        return ll


    # ----
    # Invscales model methods
    # ----

    def sample_invscales_type(self, subid):
        """
        Sample the scale parameters for each sub-stroke
        :param lib: [Library] library class instance
        :param subid: [(k,) tensor] vector of sub-stroke ids
        :return:
            invscales: [(k,) tensor] vector of scale values
        """
        if self.isunif:
            raise NotImplementedError
        # check that it is a vector
        assert len(subid.shape) == 1
        # create gamma distribution
        gamma = dist.Gamma(self.scales_con[subid], self.scales_rate[subid])
        # sample from the gamma distribution
        invscales = gamma.sample()

        return invscales

    def score_invscales_type(self, subid, invscales):
        """
        Score the log-likelihood of each sub-stroke's scale parameter
        :param lib:
        :param invscales:
        :param subid:
        :return:
        """
        if self.isunif:
            raise NotImplementedError
        # make sure these are vectors
        assert len(invscales.shape) == 1
        assert len(subid.shape) == 1
        assert len(invscales) == len(subid)
        # create gamma distribution
        gamma = dist.Gamma(self.scales_con[subid], self.scales_rate[subid])
        # score points using the gamma distribution
        ll = gamma.log_prob(invscales)

        return ll

    # ----
    # Over-arching part & relation methods
    # ----

    def sample_part_type(self, k):
        """
        See ConceptTypeDist.sample_part_type for description
        """
        # sample the number of sub-strokes
        nsub = self.sample_nsub(k)
        # sample the sequence of sub-stroke IDs
        subid = self.sample_subid(nsub)
        # sample control points for each sub-stroke in the sequence
        cpts = self.sample_shapes_type(subid)
        # sample scales for each sub-stroke in the sequence
        scales = self.sample_invscales_type(subid)
        # initialize the stroke type
        stroke = Stroke(
            subid, cpts, scales,
            sigma_shape=self.sigma_shape,
            sigma_invscale=self.sigma_invscale
        )

        return stroke

    def score_part_type(self, k, p):
        raise NotImplementedError

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

    def score_relation_type(self, prev_parts, r):
        raise NotImplementedError