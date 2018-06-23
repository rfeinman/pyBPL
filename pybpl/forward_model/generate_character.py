"""
Generate character
"""

from __future__ import print_function, division

from ..classes import MotorProgram, CPD


def generate_type(libclass, ns=None):
    """
    Generate a character type (i.e. a template for a character) by sampling
    from the prior

    :param libclass: [Library] library class instance
    :param ns: [int or tensor] number of strokes for the character
    :return:
        template: [MotorProgram] character template
    """
    if ns is None:
        # sample the number of strokes
        ns = CPD.sample_number(libclass)
    # initialize the character template
    template = MotorProgram(ns)
    # for each stroke, sample stroke parameters
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

def generate_token(libclass, template):
    """
    Given a character type (template), generate a token (i.e. and exemplar of
    the character) by sampling from the prior

    :param libclass: [Library] library class instance
    :param template: [MotorProgram] character template
    :return:
        image: [(m,n) tensor] the character sample
    """
    M = MotorProgram(template)
    #sample stroke params
    for sid in range(M.ns):
        if M.S[sid].R.rtype == 'mid':
            M.S[sid].R.eval_spot_token = CPD.sample_relation_token(
                libclass, M.S[sid].R.eval_spot_type
            )
        M.S[sid].pos_token = CPD.sample_position(
            libclass, M.S[sid].R, M.S[:sid]
        )
        M.S[sid].shapes_token = CPD.sample_shape_token(
            libclass, M.S[sid].shapes_type
        )
        M.S[sid].invscales_token = CPD.sample_invscale_token(
            libclass, M.S[sid].invscales_type
        )
    # sample affine warp
    M.A = CPD.sample_affine(libclass)

    # set rendering parameters to minimum noise
    M.blur_sigma = template.parameters.min_blur_sigma
    M.epsilon = template.parameters.min_epsilon

    # sample rendering parameters
    #M.blur_sigma = CPD.sample_image_blur(template.parameters)
    #M.epsilon = CPD.sample_image_noise(template.parameters)

    # sample the image
    M.I = CPD.sample_image(M.pimg)
    image = M.I

    return image

def generate_character(libclass, ns=None):
    template = generate_type(libclass, ns)
    character = generate_token(libclass, template)

    return character