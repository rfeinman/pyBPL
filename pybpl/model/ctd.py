"""
Concept type distributions for sampling concept types from pre-specified
type distributions.
"""
from __future__ import division, print_function
from abc import ABCMeta, abstractmethod
import warnings
import torch
import torch.distributions as dist

from ..library import Library
from ..relation import (Relation, RelationIndependent, RelationAttach,
                       RelationAttachAlong)
from ..part import Stroke
from ..concept import Concept, Character
from ..splines import bspline_gen_s

# list of acceptable dtypes for 'k' parameter
int_types = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]




class ConceptTypeDist(object):
    """
    Abstract base class for concept type distributions. 'ConceptTypeDist'
    defines the prior distribution P(Type)
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def sample_k(self):
        pass

    @abstractmethod
    def score_k(self, k):
        pass

    @abstractmethod
    def sample_part_type(self, k):
        pass

    @abstractmethod
    def score_part_type(self, p, k):
        pass

    @abstractmethod
    def sample_relation_type(self, prev_parts):
        pass

    @abstractmethod
    def score_relation_type(self, r, prev_parts):
        pass

    def sample_type(self, k=None):
        """
        Sample a concept type from the prior

        Parameters
        ----------
        k : int or tensor
            scalar; the number of parts to use

        Returns
        -------
        c : Concept
            concept type sample
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

        # initialize part and relation type lists
        P = []
        R = []
        # for each part, sample part parameters
        for _ in range(k):
            # sample the part type
            p = self.sample_part_type(k)
            # sample the relation type
            r = self.sample_relation_type(P)
            # append to the lists
            P.append(p)
            R.append(r)
        # create the concept type, i.e. a motor program for sampling
        # concept tokens
        c = Concept(k, P, R)

        return c

    def score_type(self, c):
        """
        Compute the log-probability of a concept type under the prior
        $P(type) = P(k)*\prod_{i=1}^k [P(S_i)P(R_i|S_{0:i-1})]$

        Parameters
        ----------
        c : Concept
            concept type to score

        Returns
        -------
        ll : tensor
            scalar; log-probability of the concept type
        """
        assert isinstance(c, Concept)
        # score the number of parts
        ll = 0.
        ll = ll + self.score_k(c.k)
        # step through and score each part
        for i in range(c.k):
            ll = ll + self.score_part_type(c.P[i], c.k)
            ll = ll + self.score_relation_type(c.R[i], c.P[:i])

        return ll


class CharacterTypeDist(ConceptTypeDist):
    """
    A CharacterTypeDist is a probabilistic program that samples character types
    from the prior. It can also compute the log-likelihood of character types.
    'CharacterTypeDist' defines the prior distribution P(Type)

    Parameters
    ----------
    lib : Library
        library instance
    """
    __relation_categories = ['unihist', 'start', 'end', 'mid']

    def __init__(self, lib):
        super(CharacterTypeDist, self).__init__()
        assert isinstance(lib, Library)
        self.lib = lib
        # distribution of 'k' (number of strokes)
        assert len(lib.pkappa.shape) == 1
        self.kappa = dist.Categorical(probs=lib.pkappa)
        # distribution of relation categories
        self.rel_mixdist = dist.Categorical(probs=lib.rel['mixprob'])
        # invscales distribution
        scales_theta = lib.scale['theta']
        assert len(scales_theta.shape) == 2
        self.scales_con = scales_theta[:,0] # gamma concentration
        # NOTE: PyTorch gamma dist uses rate parameter, which is inv of scale
        self.scales_rate = 1/scales_theta[:,1] # gamma rate


    # ----
    # Stroke count model methods
    # ----

    def sample_k(self):
        """
        Sample a stroke count from the prior

        Returns
        -------
        k : tensor
            scalar; stroke count
        """
        # sample from kappa
        # NOTE: add 1 to 0-indexed samples
        k = self.kappa.sample() + 1

        return k

    def score_k(self, k):
        """
        Compute the log-probability of the stroke count under the prior

        Parameters
        ----------
        k : tensor
            scalar; stroke count to score

        Returns
        -------
        ll : tensor
            scalar; log-probability of the stroke count
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

    def sample_nsub(self, k):
        """
        Sample a sub-stroke count

        Parameters
        ----------
        k : tensor
            scalar; stroke count

        Returns
        -------
        nsub : tensor
            scalar; sub-stroke count
        """
        # probability of each sub-stroke count, conditioned on the number of
        # strokes. NOTE: subtract 1 from stroke counts to get Python index
        pvec = self.lib.pmat_nsub[k-1]
        # make sure pvec is a vector
        assert len(pvec.shape) == 1
        # sample from the categorical distribution. Add 1 to 0-indexed sample
        nsub = dist.Categorical(probs=pvec).sample() + 1

        return nsub

    def score_nsub(self, k, nsub):
        """
        Compute the log-probability of a sub-stroke count under the prior

        Parameters
        ----------
        k : tensor
            scalar; stroke count
        nsub : tensor
            scalar; sub-stroke count

        Returns
        -------
        ll : tensor
            scalar; log-probability of the sub-stroke count
        """
        # nsub should be a scalar
        assert nsub.shape == torch.Size([])
        # collect pvec for this k
        pvec = self.lib.pmat_nsub[k-1]
        # make sure pvec is a vector
        assert len(pvec.shape) == 1
        # score using the categorical distribution
        ll = dist.Categorical(probs=pvec).log_prob(nsub-1)

        return ll

    def sample_subIDs(self, nsub):
        """
        Sample a sequence of sub-stroke IDs from the prior

        Parameters
        ----------
        nsub : tensor
            scalar; sub-stroke count

        Returns
        -------
        subid : (nsub,) tensor
            sub-stroke ID sequence
        """
        # nsub should be a scalar
        assert nsub.shape == torch.Size([])
        # set initial transition probabilities
        pT = torch.exp(self.lib.logStart)
        # sub-stroke sequence is a list
        subid = []
        # step through and sample 'nsub' sub-strokes
        for _ in range(nsub):
            # sample the sub-stroke
            ss = dist.Categorical(probs=pT).sample()
            subid.append(ss)
            # update transition probabilities; condition on previous sub-stroke
            pT = self.lib.pT(ss)
        # convert list into tensor
        subid = torch.stack(subid)

        return subid

    def score_subIDs(self, subid):
        """
        Compute the log-probability of a sub-stroke ID sequence under the prior

        Parameters
        ----------
        subid : (nsub,) tensor
            sub-stroke ID sequence

        Returns
        -------
        ll : (nsub,) tensor
            scalar; log-probability of the sub-stroke ID sequence
        """
        # set initial transition probabilities
        pT = torch.exp(self.lib.logStart)
        # initialize log-prob vector
        ll = torch.zeros(len(subid), dtype=torch.float)
        # step through sub-stroke IDs
        for i, ss in enumerate(subid):
            # add to log-prob accumulator
            ll[i] = dist.Categorical(probs=pT).log_prob(ss)
            # update transition probabilities; condition on previous sub-stroke
            pT = self.lib.pT(ss)

        return ll


    # ----
    # Shapes model methods
    # ----

    def sample_shapes_type(self, subid):
        """
        Sample the control points for each sub-stroke ID in a given sequence

        Parameters
        ----------
        subid : (nsub,) tensor
            sub-stroke ID sequence

        Returns
        -------
        shapes : (ncpt, 2, nsub) tensor
            sampled shapes of bsplines
        """
        if self.lib.isunif:
            raise NotImplementedError
        # record num control points
        ncpt = self.lib.ncpt
        # record shapes mean and covariance
        shapes_mu = self.lib.shape['mu'][subid]
        shapes_Cov = self.lib.shape['Sigma'][subid]
        # check that subid is a vector
        assert len(subid.shape) == 1
        # record vector length
        nsub = len(subid)
        # create multivariate normal distribution
        mvn = dist.MultivariateNormal(shapes_mu, shapes_Cov)
        # sample points from the multivariate normal distribution
        shapes = mvn.sample()
        # transpose axes (nsub, ncpt*2) -> (ncpt*2, nsub)
        shapes = shapes.transpose(0,1)
        # reshape tensor (ncpt*2, nsub) -> (ncpt, 2, nsub)
        shapes = shapes.view(ncpt,2,nsub)

        return shapes

    def score_shapes_type(self, subid, shapes):
        """
        Compute the log-probability of the control points for each sub-stroke
        under the prior

        Parameters
        ----------
        subid : (nsub,) tensor
            sub-stroke ID sequence
        shapes : (ncpt, 2, nsub) tensor
            shapes of bsplines

        Returns
        -------
        ll : (nsub,) tensor
            vector of log-likelihood scores
        """
        if self.lib.isunif:
            raise NotImplementedError
        # record num control points
        ncpt = self.lib.ncpt
        # record shapes mean and covariance
        shapes_mu = self.lib.shape['mu'][subid]
        shapes_Cov = self.lib.shape['Sigma'][subid]
        # check that subid is a vector
        assert len(subid.shape) == 1
        # record vector length
        nsub = len(subid)
        assert shapes.shape[-1] == nsub
        # reshape tensor (ncpt, 2, nsub) -> (ncpt*2, nsub)
        shapes = shapes.view(ncpt*2,nsub)
        # transpose axes (ncpt*2, nsub) -> (nsub, ncpt*2)
        shapes = shapes.transpose(0,1)
        # create multivariate normal distribution
        mvn = dist.MultivariateNormal(shapes_mu, shapes_Cov)
        # score points using the multivariate normal distribution
        ll = mvn.log_prob(shapes)

        return ll


    # ----
    # Invscales model methods
    # ----

    def sample_invscales_type(self, subid):
        """
        Sample the scale parameters for each sub-stroke

        Parameters
        ----------
        subid : (nsub,) tensor
            sub-stroke ID sequence

        Returns
        -------
        invscales : (nsub,) tensor
            scale values for each sub-stroke
        """
        if self.lib.isunif:
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
        Compute the log-probability of each sub-stroke's scale parameter
        under the prior

        Parameters
        ----------
        subid : (nsub,) tensor
            sub-stroke ID sequence
        invscales : (nsub,) tensor
            scale values for each sub-stroke

        Returns
        -------
        ll : (nsub,) tensor
            vector of log-likelihood scores
        """
        if self.lib.isunif:
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
        Sample a stroke type from the prior, conditioned on a number of strokes

        Parameters
        ----------
        k : tensor
            scalar; stroke count

        Returns
        -------
        p : Stroke
            part type sample
        """
        # sample the number of sub-strokes
        nsub = self.sample_nsub(k)
        # sample the sequence of sub-stroke IDs
        ids = self.sample_subIDs(nsub)
        # sample control points for each sub-stroke in the sequence
        shapes = self.sample_shapes_type(ids)
        # sample scales for each sub-stroke in the sequence
        invscales = self.sample_invscales_type(ids)
        # initialize the stroke type
        p = Stroke(nsub, ids, shapes, invscales, self.lib)

        return p

    def score_part_type(self, p, k):
        """
        Compute the log-probability of the stroke type, conditioned on a
        number of strokes, under the prior

        Parameters
        ----------
        p : Stroke
            part type to score
        k : tensor
            scalar; stroke count

        Returns
        -------
        ll : tensor
            scalar; log-probability of the stroke type
        """
        nsub_score = self.score_nsub(k, p.nsub)
        subIDs_scores = self.score_subIDs(p.ids)
        shapes_scores = self.score_shapes_type(p.ids, p.shapes)
        invscales_scores = self.score_invscales_type(p.ids, p.invscales)
        ll = nsub_score + torch.sum(subIDs_scores) + torch.sum(shapes_scores) \
             + torch.sum(invscales_scores)

        return ll

    def sample_relation_type(self, prev_parts):
        """
        Sample a relation type from the prior for the current stroke,
        conditioned on the previous strokes

        Parameters
        ----------
        prev_parts : list of Stroke
            previous part types

        Returns
        -------
        r : Relation
            relation type sample
        """
        for p in prev_parts:
            assert isinstance(p, Stroke)
        nprev = len(prev_parts)
        stroke_ix = nprev
        spatial = self.lib.Spatial
        xlim = spatial.xlim
        ylim = spatial.ylim
        # first sample the relation category
        if nprev == 0:
            category = 'unihist'
        else:
            indx = self.rel_mixdist.sample()
            category = self.__relation_categories[indx]

        # now sample the category-specific type-level parameters
        if category == 'unihist':
            data_id = torch.tensor([stroke_ix])
            gpos = spatial.sample(data_id)
            # convert (1,2) tensor to (2,) tensor
            gpos = torch.squeeze(gpos)
            r = RelationIndependent(category, gpos, xlim, ylim, self.lib)
        elif category in ['start', 'end', 'mid']:
            # sample random stroke uniformly from previous strokes. this is the
            # stroke we will attach to
            probs = torch.ones(nprev)
            attach_ix = dist.Categorical(probs=probs).sample()
            if category == 'mid':
                # sample random sub-stroke uniformly from the selected stroke
                nsub = prev_parts[attach_ix].nsub
                probs = torch.ones(nsub)
                attach_subix = dist.Categorical(probs=probs).sample()
                # sample random type-level spline coordinate
                _, lb, ub = bspline_gen_s(self.lib.ncpt, 1)
                eval_spot = dist.Uniform(lb, ub).sample()
                r = RelationAttachAlong(
                    category, attach_ix, attach_subix, eval_spot, self.lib
                )
            else:
                r = RelationAttach(category, attach_ix, self.lib)
        else:
            raise TypeError('invalid relation')

        return r

    def score_relation_type(self, r, prev_parts):
        """
        Compute the log-probability of the relation type of the current stroke
        under the prior

        Parameters
        ----------
        r : Relation
            relation type to score
        prev_parts : list of Stroke
            previous stroke types

        Returns
        -------
        ll : tensor
            scalar; log-probability of the relation type
        """
        assert isinstance(r, Relation)
        for p in prev_parts:
            assert isinstance(p, Stroke)
        nprev = len(prev_parts)
        stroke_ix = nprev
        spatial = self.lib.Spatial
        # first score the relation category
        if nprev == 0:
            ll = 0.
        else:
            ix = self.__relation_categories.index(r.category)
            ix = torch.tensor(ix, dtype=torch.long)
            ll = self.rel_mixdist.log_prob(ix)

        # now score the category-specific type-level parameters
        if r.category == 'unihist':
            data_id = torch.tensor([stroke_ix])
            # convert (2,) tensor to (1,2) tensor
            gpos = r.gpos.view(1,2)
            # score the type-level location
            ll = ll + torch.squeeze(spatial.score(gpos, data_id))
        elif r.category in ['start', 'end', 'mid']:
            # score the stroke attachment index
            probs = torch.ones(nprev)
            ll = ll + dist.Categorical(probs=probs).log_prob(r.attach_ix)
            if r.category == 'mid':
                # score the sub-stroke attachment index
                nsub = prev_parts[r.attach_ix].nsub
                probs = torch.ones(nsub)
                ll = ll + dist.Categorical(probs=probs).log_prob(r.attach_subix)
                # score the type-level spline coordinate
                _, lb, ub = bspline_gen_s(self.lib.ncpt, 1)
                ll = ll + dist.Uniform(lb, ub).log_prob(r.eval_spot)
        else:
            raise TypeError('invalid relation')

        return ll

    def sample_type(self, k=None):
        """
        Sample a character type

        Parameters
        ----------
        k : int
            optional; number of strokes for the type. If 'None' this will be
            sampled

        Returns
        -------
        c : Character
            character type

        """
        c = super(CharacterTypeDist, self).sample_type(k)
        c = Character(c.k, c.P, c.R, self.lib)

        return c