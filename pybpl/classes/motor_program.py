"""
Motor program.
"""
from __future__ import print_function, division
import copy
import torch

from pybpl.classes.stroke import Stroke


class MotorProgram(object): 
    def __init__(self, args):
        #set variables:
        self.I = []
        self.S = []
        self.parameters = []

        if isinstance(args, torch.Tensor):
            assert args.data.shape == torch.Size([])
            ns = args
            for i in range(ns):
                self.S.append(Stroke())
        elif isinstance(args, MotorProgram):
            Template = args
            for i in range(Template.ns):
                self.S.append(Stroke(Template.S[i]))
            # this might break if mcmc comes online
            self.parameters = copy.copy(Template.parameters)
        else:
            raise TypeError("Invalid constructor pararmeter; must be either a "
                            "torch.Tensor or MotorProgram")


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



