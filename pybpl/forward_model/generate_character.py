"""
Generate character
"""

from __future__ import print_function, division
import torch

from pybpl.classes import MotorProgram, CPD
from pybpl.forward_model import generate_exemplar
from pybpl.parameters import defaultps


def generate_character(libclass, ns=None):
    if ns is None:
        # since we are calling this with nsamp=1, access the 0-th element
        ns = CPD.sample_number(libclass)[0]
    template = MotorProgram(ns)
    template.parameters = defaultps()
    print('ns: %i' % ns)
    # for each stroke, sample its template
    for i in range(ns):
        # sample the relation type for this stroke
        template.S[i].R = CPD.sample_relation_type(libclass, template.S[:i])
        # sample the sequence of substrokes
        # since we use nsamp=1 (the default), access the 0-th element
        template.S[i].ids = CPD.sample_sequence(libclass, ns)[0]
        template.S[i].shapes_type = CPD.sample_shape_type(
            libclass, template.S[i].ids
        )
        template.S[i].invscales_type = CPD.sample_invscale_type(
            libclass, template.S[i].ids
        )
    return template, lambda: generate_exemplar(template, libclass)