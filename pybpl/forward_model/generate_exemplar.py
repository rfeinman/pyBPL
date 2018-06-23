"""
Generate exemplar
"""
from __future__ import division, print_function

from ..classes import MotorProgram, CPD


def generate_exemplar(template, libclass):
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

    return M, template
