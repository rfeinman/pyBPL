"""
Concept type distributions for sampling concept types from pre-specified
type distributions.
"""
from __future__ import division, print_function
import warnings
import torch
import torch.distributions as dist

from ..library import Library
from ..relation import (RelationType, RelationIndependent, RelationAttach,
                       RelationAttachAlong)
from ..part import StrokeType
from ..concept import CharacterType
from ..splines import bspline_gen_s
import pyprob
import pybpl

# list of acceptable dtypes for 'k' parameter
int_types = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]


class CharacterTypeDist:
    """
    Defines the prior distribution P(Type) for character types

    Parameters
    ----------
    lib : Library
        library instance
    """

    def __init__(self, lib):
        assert isinstance(lib, Library)
        self.stroke_type_dist = StrokeTypeDist(lib)
        self.relation_type_dist = RelationTypeDist(lib)

        # distribution of 'k' (number of strokes)
        assert len(lib.pkappa.shape) == 1

        # PYPROB
        self.kappa = pyprob.distributions.Categorical(probs=lib.pkappa)

        # ORIGINAL
        # self.kappa = dist.Categorical(probs=lib.pkappa)

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
        # Categorical p(kappa)
        # PYPROB
        k = int(pyprob.sample(self.kappa, address='kappa')) + 1

        # ORIGINAL
        # k = self.kappa.sample() + 1

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
        c : CharacterType
            character type

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

        # initialize stroke and relation type lists
        stroke_types = []
        relation_types = []
        # for each part, sample part parameters
        for _ in range(k):
            # sample the part type
            stroke_type = self.stroke_type_dist.sample_stroke_type(k)
            # sample the relation type
            relation_type = self.relation_type_dist.sample_relation_type(
                stroke_types)
            # append to the lists
            stroke_types.append(stroke_type)
            relation_types.append(relation_type)
        # create the concept type, i.e. a motor program for sampling
        # concept tokens
        character_type = CharacterType(k, stroke_types, relation_types)

        return character_type

    def score_type(self, character_type):
        """
        Compute the log-probability of a concept type under the prior
        $P(type) = P(k)*\prod_{i=1}^k [P(S_i)P(R_i|S_{0:i-1})]$

        Parameters
        ----------
        ctype : CharacterType
            concept type to score

        Returns
        -------
        ll : tensor
            scalar; log-probability of the concept type
        """
        assert isinstance(character_type, CharacterType)
        # score the number of parts
        ll = 0.
        ll = ll + self.score_k(character_type.k)
        # step through and score each part
        for i in range(character_type.k):
            ll = ll + self.stroke_type_dist.score_stroke_type(
                character_type.k, character_type.stroke_types[i])
            ll = ll + self.relation_type_dist.score_relation_type(
                character_type.stroke_types[:i],
                character_type.relation_types[i])

        return ll


class StrokeTypeDist:
    def __init__(self, lib):
        # is uniform
        self.isunif = lib.isunif
        # number of control points
        self.ncpt = lib.ncpt
        # sub-stroke count distribution
        self.pmat_nsub = lib.pmat_nsub
        # sub-stroke id distribution
        self.logStart = lib.logStart
        self.pT = lib.pT
        # shapes distribution
        self.shapes_mu = lib.shape['mu']
        self.shapes_Cov = lib.shape['Sigma']
        # invscales distribution
        scales_theta = lib.scale['theta']
        assert len(scales_theta.shape) == 2
        self.scales_con = scales_theta[:,0]  # gamma concentration
        # NOTE: PyTorch gamma dist uses rate parameter, which is inv of scale
        self.scales_rate = 1 / scales_theta[:,1]  # gamma rate

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
        pvec = self.pmat_nsub[k-1]
        # make sure pvec is a vector
        assert len(pvec.shape) == 1
        # sample from the categorical distribution. Add 1 to 0-indexed sample
        # Categorical p(n_i | i)
        # PYPROB
        nsub_minus_one = pyprob.sample(pyprob.distributions.Categorical(probs=pvec), address='nsub_minus_one')
        nsub = nsub_minus_one.long() + 1

        # ORIGINAL
        # nsub = dist.Categorical(probs=pvec).sample() + 1

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
        pvec = self.pmat_nsub[k-1]
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
        pT = torch.exp(self.logStart)
        # sub-stroke sequence is a list
        subid = []
        # step through and sample 'nsub' sub-strokes
        for _ in range(nsub):
            # sample the sub-stroke
            # Categorical p(z_ij | z_i(j - 1))
            # PYPROB
            ss = pyprob.sample(pyprob.distributions.Categorical(probs=pT),
                               address='ss').long()

            # ORIGINAL
            # ss = dist.Categorical(probs=pT).sample()
            subid.append(ss)
            # update transition probabilities; condition on previous sub-stroke
            pT = self.pT(ss)
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
        pT = torch.exp(self.logStart)
        # initialize log-prob vector
        ll = torch.zeros(len(subid), dtype=torch.float)
        # step through sub-stroke IDs
        for i, ss in enumerate(subid):
            # add to log-prob accumulator
            ll[i] = dist.Categorical(probs=pT).log_prob(ss)
            # update transition probabilities; condition on previous sub-stroke
            pT = self.pT(ss)

        return ll

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
        if self.isunif:
            raise NotImplementedError
        # check that subid is a vector
        assert len(subid.shape) == 1
        # record vector length
        nsub = len(subid)
        # MVN p(x_ij | z_ij)
        # PYPROB
        # TODO implement MVN in pyprob or convert this to indep Normals
        normal = pyprob.distributions.Normal(
            self.shapes_mu[subid],
            torch.sqrt(torch.einsum('ijk->ij', self.shapes_Cov[subid])))
        shapes = pyprob.sample(normal,
                               control=pybpl.TRAIN_NON_CATEGORICALS,
                               address='shapes')

        # ORIGINAL
        # create multivariate normal distribution
        # mvn = dist.MultivariateNormal(
        #     self.shapes_mu[subid], self.shapes_Cov[subid]
        # )
        # sample points from the multivariate normal distribution
        # shapes = mvn.sample()

        # transpose axes (nsub, ncpt*2) -> (ncpt*2, nsub)
        shapes = shapes.transpose(0,1)
        # reshape tensor (ncpt*2, nsub) -> (ncpt, 2, nsub)
        shapes = shapes.view(self.ncpt,2,nsub)

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
        if self.isunif:
            raise NotImplementedError
        # check that subid is a vector
        assert len(subid.shape) == 1
        # record vector length
        nsub = len(subid)
        assert shapes.shape[-1] == nsub
        # reshape tensor (ncpt, 2, nsub) -> (ncpt*2, nsub)
        shapes = shapes.view(self.ncpt*2,nsub)
        # transpose axes (ncpt*2, nsub) -> (nsub, ncpt*2)
        shapes = shapes.transpose(0,1)
        # create multivariate normal distribution
        mvn = dist.MultivariateNormal(
            self.shapes_mu[subid], self.shapes_Cov[subid]
        )
        # score points using the multivariate normal distribution
        ll = mvn.log_prob(shapes)

        return ll

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
        if self.isunif:
            raise NotImplementedError
        # check that it is a vector
        assert len(subid.shape) == 1

        # Gamma p(y_ij | z_ij)
        # PYPROB
        # TODO: improve gamma proposal in pyprob
        nsub = len(subid)
        invscales = torch.zeros(nsub)
        for i in range(nsub):
            gamma = pyprob.distributions.Gamma(self.scales_con[subid[i]],
                                               self.scales_rate[subid[i]])
            invscales[i] = pyprob.sample(gamma,
                                         control=pybpl.TRAIN_NON_CATEGORICALS,
                                         address='invscales_i')

        # ORIGINAL
        # create gamma distribution
        # gamma = dist.Gamma(self.scales_con[subid], self.scales_rate[subid])
        # sample from the gamma distribution
        # invscales = gamma.sample()

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

    def sample_stroke_type(self, k):
        """
        Sample a stroke type from the prior, conditioned on a stroke count

        Parameters
        ----------
        k : tensor
            scalar; stroke count

        Returns
        -------
        p : StrokeType
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
        p = StrokeType(nsub, ids, shapes, invscales)

        return p

    def score_stroke_type(self, k, ptype):
        """
        Compute the log-probability of the stroke type, conditioned on a
        stroke count, under the prior

        Parameters
        ----------
        k : tensor
            scalar; stroke count
        ptype : StrokeType
            part type to score

        Returns
        -------
        ll : tensor
            scalar; log-probability of the stroke type
        """
        nsub_score = self.score_nsub(k, ptype.nsub)
        subIDs_scores = self.score_subIDs(ptype.ids)
        shapes_scores = self.score_shapes_type(ptype.ids, ptype.shapes)
        invscales_scores = self.score_invscales_type(ptype.ids, ptype.invscales)
        ll = nsub_score + torch.sum(subIDs_scores) + torch.sum(shapes_scores) \
             + torch.sum(invscales_scores)

        return ll


class RelationTypeDist(object):
    __relation_categories = ['unihist', 'start', 'end', 'mid']
    def __init__(self, lib):
        self.ncpt = lib.ncpt
        self.Spatial = lib.Spatial
        # distribution of relation categories
        # PYPROB
        self.rel_mixdist = pyprob.distributions.Categorical(
            probs=lib.rel['mixprob'])

        # ORIGINAL
        # self.rel_mixdist = dist.Categorical(probs=lib.rel['mixprob'])

    def sample_relation_type(self, prev_parts):
        """
        Sample a relation type from the prior for the current stroke,
        conditioned on the previous strokes

        Parameters
        ----------
        prev_parts : list of StrokeType
            previous part types

        Returns
        -------
        r : RelationType
            relation type sample
        """
        for p in prev_parts:
            assert isinstance(p, StrokeType)
        nprev = len(prev_parts)
        stroke_ix = nprev
        # first sample the relation category
        if nprev == 0:
            category = 'unihist'
        else:
            # Categorical
            # PYPROB
            indx = pyprob.sample(self.rel_mixdist, address='indx').long()

            # ORIGINAL
            # indx = self.rel_mixdist.sample()
            category = self.__relation_categories[indx]

        # now sample the category-specific type-level parameters
        if category == 'unihist':
            data_id = torch.tensor([stroke_ix])
            gpos = self.Spatial.sample(data_id)
            # convert (1,2) tensor to (2,) tensor
            gpos = torch.squeeze(gpos)
            r = RelationIndependent(
                category, gpos, self.Spatial.xlim, self.Spatial.ylim
            )
        elif category in ['start', 'end', 'mid']:
            # sample random stroke uniformly from previous strokes. this is the
            # stroke we will attach to
            probs = torch.ones(nprev)

            # Categorical p(u_i | zeta_i)
            # PYPROB
            attach_ix = pyprob.sample(pyprob.distributions.Categorical(
                probs=probs), address='attach_ix').long()

            # ORIGINAL
            # attach_ix = dist.Categorical(probs=probs).sample()
            if category == 'mid':
                # sample random sub-stroke uniformly from the selected stroke
                nsub = prev_parts[attach_ix].nsub
                probs = torch.ones(nsub)

                # Categorical p(v_i | zeta_i)
                # PYPROB
                attach_subix = pyprob.sample(pyprob.distributions.Categorical(
                    probs=probs), address='attach_subix').long()

                # ORIGINAL
                # attach_subix = dist.Categorical(probs=probs).sample()

                # sample random type-level spline coordinate
                _, lb, ub = bspline_gen_s(self.ncpt, 1)

                # Uniform p(tau_i | zeta_i)
                # PYPROB
                eval_spot = pyprob.sample(pyprob.distributions.Uniform(lb, ub),
                                          control=pybpl.TRAIN_NON_CATEGORICALS,
                                          address='eval_spot')

                # ORIGINAL
                # eval_spot = dist.Uniform(lb, ub).sample()

                r = RelationAttachAlong(
                    category, attach_ix, attach_subix, eval_spot, self.ncpt
                )
            else:
                r = RelationAttach(category, attach_ix)
        else:
            raise TypeError('invalid relation')

        return r

    def score_relation_type(self, prev_parts, r):
        """
        Compute the log-probability of the relation type of the current stroke
        under the prior

        Parameters
        ----------
        prev_parts : list of StrokeType
            previous stroke types
        r : RelationType
            relation type to score

        Returns
        -------
        ll : tensor
            scalar; log-probability of the relation type
        """
        assert isinstance(r, RelationType)
        for p in prev_parts:
            assert isinstance(p, StrokeType)
        nprev = len(prev_parts)
        stroke_ix = nprev
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
            ll = ll + torch.squeeze(self.Spatial.score(gpos, data_id))
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
                _, lb, ub = bspline_gen_s(self.ncpt, 1)
                ll = ll + dist.Uniform(lb, ub).log_prob(r.eval_spot)
        else:
            raise TypeError('invalid relation')

        return ll
