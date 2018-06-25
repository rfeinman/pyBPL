"""
Relation model (R)
"""
from __future__ import division, print_function
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.uniform import Uniform

from ..concept.relation import (RelationIndependent, RelationAttach,
                                    RelationAttachAlong)
from ..splines import bspline_gen_s


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
    mixprob = lib.rel['mixprob']
    sigma_x = lib.rel['sigma_x']
    sigma_y = lib.rel['sigma_y']
    sigma_attach = lib.tokenvar['sigma_attach']
    if nprev == 0:
        indx = torch.tensor(0, dtype=torch.int64, requires_grad=True)
    else:
        indx = Categorical(probs=mixprob).sample()

    rtype = types[indx]

    if rtype == 'unihist':
        data_id = torch.tensor([stroke_num])
        gpos = lib.Spatial.sample(data_id)
        # create relation
        r = RelationIndependent(rtype, nprev, sigma_x, sigma_y, gpos)
    elif rtype in ['start', 'end']:
        # sample random attach spot uniformly
        probs = torch.ones(nprev, requires_grad=True)
        attach_spot = Categorical(probs=probs).sample()
        # create relation
        r = RelationAttach(rtype, nprev, sigma_x, sigma_y, attach_spot)
    elif rtype == 'mid':
        # sample random attach spot uniformly
        probs = torch.ones(nprev, requires_grad=True)
        attach_spot = Categorical(probs=probs).sample()
        # sample random subid spot uniformly
        nsub = prev_strokes[attach_spot].nsub
        probs = torch.ones(nsub, requires_grad=True)
        subid_spot = Categorical(probs=probs).sample()
        # sample eval_spot_type
        _, lb, ub = bspline_gen_s(ncpt, 1)
        eval_spot_type = Uniform(lb, ub).sample()
        # create relation
        r = RelationAttachAlong(
            rtype, nprev, sigma_x, sigma_y, sigma_attach, attach_spot,
            subid_spot, ncpt, eval_spot_type
        )

    else:
        raise TypeError('invalid relation')

    return r