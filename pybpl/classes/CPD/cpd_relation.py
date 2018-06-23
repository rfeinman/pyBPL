"""
Relation model (R)
"""
from __future__ import division, print_function
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

from ..relations import (RelationIndependent, RelationAttach,
                        RelationAttachAlong)
from ...splines import bspline_gen_s

def sample_relation_type(libclass, prev_strokes):
    """
    Sample a relation type for this stroke

    :param libclass: [Library] library class instance
    :param prev_strokes: [list of Strokes] list of previous strokes
    :return:
        R: [Relation] relation
    """
    nprev = len(prev_strokes)
    stroke_num = nprev + 1
    types = ['unihist', 'start', 'end', 'mid']
    ncpt = libclass.ncpt
    if nprev == 0:
        indx = torch.tensor(0, dtype=torch.int64, requires_grad=True)
    else:
        indx = Categorical(probs=libclass.rel['mixprob']).sample()

    rtype = types[indx]

    if rtype == 'unihist':
        data_id = torch.tensor([stroke_num])
        gpos = libclass.Spatial.sample(data_id)
        R = RelationIndependent(rtype, nprev, gpos)
    elif rtype in ['start', 'end']:
        # sample random attach spot uniformly
        probs = torch.ones(nprev, requires_grad=True)
        attach_spot = Categorical(probs=probs).sample()
        R = RelationAttach(rtype, nprev, attach_spot)
    elif rtype == 'mid':
        # sample random attach spot uniformly
        probs = torch.ones(nprev, requires_grad=True)
        attach_spot = Categorical(probs=probs).sample()
        # sample random subid spot uniformly
        nsub = prev_strokes[attach_spot].nsub
        probs = torch.ones(nsub, requires_grad=True)
        subid_spot = Categorical(probs=probs).sample()
        R = RelationAttachAlong(
            rtype, nprev, attach_spot, nsub, subid_spot, ncpt
        )
        # set R.eval_spot_type
        _, lb, ub = bspline_gen_s(ncpt, 1)
        R.eval_spot_type = Uniform(lb, ub).sample()
    else:
        raise TypeError('invalid relation')

    return R

def sample_relation_token(libclass, eval_spot_type):
    loc = torch.tensor(0.)
    scale = libclass.tokenvar['sigma_attach']
    eval_spot_token = eval_spot_type + Normal(loc, scale).sample()

    ncpt = libclass.ncpt
    _, lb, ub = bspline_gen_s(ncpt, 1);  # need to fix
    while eval_spot_token.data[0] < lb or eval_spot_token.data[0] > ub:
        print("lb:", lb)
        print("ub:", ub)
        print("eval_spot_token.data[0]:", eval_spot_token.data[0])
        eval_spot_token = eval_spot_type + sigma_attach.numpy() * \
                                           pyro.sample('randn_for_rtoken',
                                                       dist.normal,
                                                       Variable(torch.zeros(1)),
                                                       Variable(torch.ones(1)))

    return eval_spot_token

def score_relation_token(libclass, eval_spot_token, eval_spot_type):
    raise NotImplementedError