"""
Generate character
"""

from __future__ import print_function, division

from pybpl.classes import MotorProgram, CPD
from pybpl.forward_model import generate_exemplar
from pybpl.parameters import defaultps


def generate_character(libclass, ns=None):
    if ns is None:
        numstrokes = CPD.sample_number(libclass)[0]
        ns = numstrokes.item()
    template = MotorProgram(ns)
    template.parameters = defaultps()
    print('ns: %i' % ns)
    # for each stroke, sample its template
    for i in range(ns):
        # this needs to be checked
        template.S[i].R = CPD.sample_relation_type(libclass, template.S[0:i])
        template.S[i].ids = CPD.sample_sequence(libclass, ns)[0]
        template.S[i].shapes_type = CPD.sample_shape_type(
            libclass, template.S[i].ids
        )
        template.S[i].invscales_type = CPD.sample_invscale_type(
            libclass, template.S[i].ids
        )
    return template, lambda: generate_exemplar(template, libclass)