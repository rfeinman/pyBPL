"""
Generate character
"""

from __future__ import print_function, division

from ..classes import MotorProgram, CPD


def generate_character(libclass, ns=None):
    if ns is None:
        # sample the number of strokes
        ns = CPD.sample_number(libclass)
    template = MotorProgram(ns)
    # for each stroke, sample its template
    for sid in range(ns):
        # sample the number of sub-strokes
        nsub = CPD.sample_nsub(libclass, ns)
        # sample the sub-stroke sequence
        template.S[sid].ids = CPD.sample_sequence(libclass, nsub)
        # sample control points for each sub-stroke in the sequence
        template.S[sid].shapes_type = CPD.sample_shape_type(
            libclass, template.S[sid].ids
        )
        # sample scales for each sub-stroke in the sequence
        template.S[sid].invscales_type = CPD.sample_invscale_type(
            libclass, template.S[sid].ids
        )
        # sample the relation of this stroke to previous strokes
        template.S[sid].R = CPD.sample_relation_type(libclass, template.S[:sid])

    return template