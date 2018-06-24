"""
Relation model (R)
"""
from __future__ import division, print_function
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.uniform import Uniform
import torch.distributions as dist

from pybpl.concept.relation import (RelationIndependent, RelationAttach,
                                    RelationAttachAlong)
from ...splines import bspline_gen_s


def __get_dist(eval_spot_type, sigma_attach):
    norm = dist.normal.Normal(eval_spot_type, sigma_attach)

    return norm

def sample_relation_type(lib, prev_strokes):
    """
    Sample a relation type for this stroke

    :param lib: [Library] library class instance
    :param prev_strokes: [list of Strokes] list of previous strokes
    :return:
        R: [Relation] relation
    """
    nprev = len(prev_strokes)
    stroke_num = nprev + 1
    types = ['unihist', 'start', 'end', 'mid']
    ncpt = lib.ncpt
    if nprev == 0:
        indx = torch.tensor(0, dtype=torch.int64, requires_grad=True)
    else:
        indx = Categorical(probs=lib.rel['mixprob']).sample()

    rtype = types[indx]

    if rtype == 'unihist':
        data_id = torch.tensor([stroke_num])
        gpos = lib.Spatial.sample(data_id)
        r = RelationIndependent(rtype, nprev, gpos)
    elif rtype in ['start', 'end']:
        # sample random attach spot uniformly
        probs = torch.ones(nprev, requires_grad=True)
        attach_spot = Categorical(probs=probs).sample()
        r = RelationAttach(rtype, nprev, attach_spot)
    elif rtype == 'mid':
        # sample random attach spot uniformly
        probs = torch.ones(nprev, requires_grad=True)
        attach_spot = Categorical(probs=probs).sample()
        # sample random subid spot uniformly
        nsub = prev_strokes[attach_spot].nsub
        probs = torch.ones(nsub, requires_grad=True)
        subid_spot = Categorical(probs=probs).sample()
        r = RelationAttachAlong(
            rtype, nprev, attach_spot, nsub, subid_spot, ncpt
        )
        # set R.eval_spot_type
        _, lb, ub = bspline_gen_s(ncpt, 1)
        r.eval_spot_type = Uniform(lb, ub).sample()
    else:
        raise TypeError('invalid relation')

    return r

def sample_relation_token(lib, eval_spot_type):
    """
    TODO

    :param lib:
    :param eval_spot_type:
    :return:
    """
    norm = __get_dist(eval_spot_type, lib.tokenvar['sigma_attach'])
    score = torch.tensor(-np.inf)
    while np.isinf(score):
        eval_spot_token = norm.sample()
        score = score_relation_token(lib, eval_spot_token, eval_spot_type)

    return eval_spot_token

def score_relation_token(lib, eval_spot_token, eval_spot_type):
    """
    TODO

    :param lib:
    :param eval_spot_token:
    :param eval_spot_type:
    :return:
    """
    assert type(eval_spot_token) in [int, float] or \
           (type(eval_spot_token) == torch.Tensor and
            eval_spot_token.shape == torch.Size([]))
    assert eval_spot_type is not None
    ncpt = lib.ncpt
    _, lb, ub = bspline_gen_s(ncpt, 1)
    if eval_spot_token < lb or eval_spot_token > ub:
        ll = torch.tensor(-np.inf)
        return ll
    norm = __get_dist(eval_spot_type, lib.tokenvar['sigma_attach'])
    ll = norm.log_prob(eval_spot_token)

    # correction for bounds
    p_within = norm.cdf(ub) - norm.cdf(lb)
    ll = ll - torch.log(p_within)

    return ll