"""
Motor program class definition
"""
from __future__ import print_function, division
import copy
import torch

from .stroke import Stroke
from ..parameters import defaultps


class MotorProgram(object):
    po = {'epsilon', 'blur_sigma', 'A'}

    def __init__(self, arg):
        self.I = []
        self.S = []
        self.parameters = []
        self.epsilon = None
        self.blur_sigma = None
        self.A = []

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
                "Invalid constructor; must be either a torch.Tensor or "
                "MotorProgram"
            )


    #other methods:

    #get number of strokes
    @property
    def ns(self):
        return len(self.S)

    #get motor??

    #get.motor(this)
    #get.motor_warped(this)
    #get.pimg(this)
    #get.ink_off_page(this)
    #get.cache_grand_current(this)
    #has_relations(this,last_sid)
    #clear_relations(this)

    #[pimg, ink_off_page] = apply_render(this) -- this is the thing that
    # needs to be differentiable

    def apply_render(self):
        ping, ink_off_page = rendering.motor_to_pimg(self)
        return pimg, ink_off_page

    @property
    def pimg(self):
        return self.apply_render()[0] #hopefully this will work

    #this is probably not the correct way to do this. May mess with rendering.
    @property
    def ink_off_page(self):
        return self.apply_render()[1]



