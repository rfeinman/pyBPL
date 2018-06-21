"""
Motor program class definition
"""
from __future__ import print_function, division
import copy
import torch

from . import Stroke, UtilMP
from ..parameters import defaultps
from .. import rendering


class MotorProgram(object):
    __po = {'epsilon', 'blur_sigma', 'A'}

    def __init__(self, arg):
        self.I = None
        self.S = []
        self.parameters = None
        self.epsilon = None
        self.blur_sigma = None
        self.A = None

        if isinstance(arg, torch.Tensor):
            assert arg.shape == torch.Size([]), \
                'Tensor parameter must be a scalar'
            for _ in range(arg):
                self.S.append(Stroke())
            self.parameters = defaultps()
        elif isinstance(arg, MotorProgram):
            template = arg
            for i in range(template.ns):
                self.S.append(Stroke(template.S[i]))
            # this might break if mcmc comes online
            self.parameters = copy.copy(template.parameters)
        else:
            raise TypeError(
                "Invalid constructor; must be either a torch.Tensor or a "
                "MotorProgram"
            )

    @property
    def ns(self):
        # get number of strokes
        return len(self.S)

    @property
    def motor(self):
        # get motor trajectories for each stroke
        return [stroke.motor for stroke in self.S]

    @property
    def motor_warped(self):
        return self.__apply_warp()

    @property
    def pimg(self):
        # get probability map of an image
        pimg, _ = self.__apply_render()

        return pimg

    @property
    def ink_off_page(self):
        _, ink_off_page = self.__apply_render()

        return ink_off_page

    def has_relations(self, list_sid=None):
        if list_sid is None:
            list_sid = range(self.ns)
        present = [self.S[i].R is not None for i in list_sid]
        assert np.all(present) or not np.any(present), \
            'error: all relations should be present or not'
        out = np.all(present)

        return out

    def clear_relations(self):
        for i in range(self.ns):
            self.S[i].R = None
        return

    def clear_shapes_type(self):
        for i in range(self.ns):
            self.S[i].shapes_type = None

    def istied(self, varargin):
        raise NotImplementedError("'istied' method not yet implemented")

    def __apply_warp(self):
        motor_unwarped = self.motor
        if self.A is None:
            return motor_unwarped
        else:
            raise NotImplementedError(
                "'apply_warp' method not yet implemented."
            )

    def __apply_render(self):
        """
        TODO - this needs to be differentiable
        motor
        :return:
            pimg: TODO
            ink_off_page: TODO
        """
        motor_warped = self.__aply_warp()
        flat_warped = UtilMP.flatten_substrokes(motor_warped)
        pimg, ink_off_page = rendering.render_image(
            motor_warped, self.epsilon, self.blur_sigma, MP.parameters
        )

        return pimg, ink_off_page


