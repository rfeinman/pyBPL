from __future__ import division, print_function
import torch
import torch.distributions as dist

from ..concept.ctd import ConceptTypeDist
from ..concept.relation import (RelationIndependent, RelationAttach,
                                RelationAttachAlong)
from ..splines import bspline_gen_s

# list of acceptable dtypes for 'np' parameter
int_types = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]


class CharacterTypeDist(ConceptTypeDist):
    relation_types = ['unihist', 'start', 'end', 'mid']

    def __init__(self, lib):
        super(CharacterTypeDist, self).__init__(lib)
        assert len(lib.pkappa.shape) == 1
        # num control points
        self.ncpt = lib.ncpt
        # distribution of 'np' (num parts)
        self.kappa = dist.Categorical(probs=lib.pkappa)
        # distribution of unihist relation positions
        self.Spatial = lib.Spatial
        # distribution of relation types
        self.rel_mixdist = dist.Categorical(probs=lib.rel['mixprob'])
        # token-level variance parameters for relations
        self.rel_pos_dist = get_rel_pos_dist(
            lib.rel['sigma_x'], lib.rel['sigma_y']
        )
        self.rel_sigma_attach = lib.tokenvar['sigma_attach']

    def sample_np(self, nsamp=1):
        """
        See ConceptTypeDist.sample_np for description
        """
        # sample from kappa
        # NOTE: add 1 to 0-indexed samples
        np = self.kappa.sample(torch.Size([nsamp])) + 1
        # make sure np is a vector
        assert len(np.shape) == 1
        # convert vector to scalar if nsamp=1
        np = torch.squeeze(np)

        return np

    def score_np(self, np):
        """
        See ConceptTypeDist.score_np for description
        """
        # check if any values are out of bounds
        out_of_bounds = np > len(self.kappa.probs)
        if out_of_bounds.any():
            ll = torch.tensor(-float('Inf'))
            return ll
        # score points using kappa
        # NOTE: subtract 1 to get 0-indexed samples
        ll = self.kappa.log_prob(np - 1)

        return ll

    def sample_part_type(self, np):
        """
        See ConceptTypeDist.sample_part_type for description
        """

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


def get_rel_pos_dist(sigma_x, sigma_y):
    """
    The token-level position distribution

    :param sigma_x:
    :param sigma_y:
    :return:
    """
    Cov = torch.eye(2)
    Cov[0,0] = sigma_x
    Cov[1,1] = sigma_y
    pos_dist = dist.multivariate_normal.MultivariateNormal(
        torch.zeros(2), Cov
    )

    return pos_dist