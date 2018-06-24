"""
Motor program class definition
"""
from __future__ import print_function, division
import torch

from .stroke import Stroke
from . import CPD
from .concept_type import ConceptType
from . import UtilMP
from ..parameters import defaultps
from ..rendering import render_image


class MotorProgram(object):

    def __init__(self, ctype):
        assert isinstance(ctype, ConceptType)
        self.ctype = ctype
        self.parameters = defaultps()

    def sample_token(self, libclass):
        """
        Sample a token from the motor program

        :param libclass: [Library] library class instance
        :return:
            image: [(m,n) tensor] token (image)
        """
        # sample the token-level stroke params
        strokes = []
        for stype, r in zip(self.ctype.S, self.ctype.R):
            # TODO - need to do something about updating eval_spot_type/token?
            if r.type == 'mid':
                r.eval_spot_token = CPD.sample_relation_token(libclass, r.eval_spot_type)
            pos_token = CPD.sample_position(libclass, r, strokes)
            shapes_token = CPD.sample_shape_token(libclass, stype.shapes_type)
            invscales_token = CPD.sample_invscale_token(libclass, stype.invscales_type)
            s = Stroke(stype, pos_token, shapes_token, invscales_token)
            strokes.append(s)

        # sample affine warp
        affine = CPD.sample_affine(libclass)

        # set rendering parameters to minimum noise
        blur_sigma = self.parameters.min_blur_sigma
        epsilon = self.parameters.min_epsilon
        # self.blur_sigma = CPD.sample_image_blur(self.parameters)
        # self.epsilon = CPD.sample_image_noise(self.parameters)

        # get probability map of an image
        pimg, _ = self.__apply_render(strokes, affine, epsilon, blur_sigma)
        # sample the image
        image = CPD.sample_image(pimg)

        return image

    @property
    def ns(self):
        # get number of strokes
        return len(self.ctype.S)

    def has_relations(self, list_sid=None):
        if list_sid is None:
            list_sid = range(self.ns)
        present = [self.ctype.R[sid] is not None for sid in list_sid]
        assert all(present) or not any(present), \
            'error: all relations should be present or not'
        out = all(present)

        return out

    def istied(self, varargin):
        raise NotImplementedError

    def __apply_warp(self, strokes, affine):
        motor_unwarped = [stroke.motor for stroke in strokes]
        if affine is None:
            motor_warped = motor_unwarped
        else:
            raise NotImplementedError(
                "'apply_warp' method not yet implemented."
            )

        return motor_warped

    def __apply_render(self, strokes, affine, epsilon, blur_sigma):
        motor_warped = self.__apply_warp(strokes, affine)
        flat_warped = UtilMP.flatten_substrokes(motor_warped)
        pimg, ink_off_page = render_image(
            flat_warped, epsilon, blur_sigma, self.parameters
        )

        return pimg, ink_off_page


