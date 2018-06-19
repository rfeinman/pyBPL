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
        # sample the number of strokes
        ns = CPD.sample_number(libclass)
    template = MotorProgram(ns)
    template.parameters = defaultps()
    print('ns: %i' % ns)
    # for each stroke, sample its template
    for i in range(ns):
        # sample the number of sub-strokes
        nsub = CPD.sample_nsub(libclass, ns)
        # sample the sub-stroke sequence. Access 0-th element since nsamp=1
        template.S[i].ids = CPD.sample_sequence(libclass, nsub)[0]
        # sample control points for each sub-stroke in the sequence
        template.S[i].shapes_type = CPD.sample_shape_type(
            libclass, template.S[i].ids
        )
        # sample scales for each sub-stroke in the sequence
        template.S[i].invscales_type = CPD.sample_invscale_type(
            libclass, template.S[i].ids
        )
        # sample the relation of this stroke to previous strokes
        template.S[i].R = CPD.sample_relation_type(libclass, template.S[:i])

    return template, lambda: generate_exemplar(template, libclass)