"""
Relation class definitions
"""
from __future__ import print_function, division

from ..splines import bspline_eval


class Relation(object):
    types_allowed = ['unihist', 'start', 'end', 'mid']

    def __init__(self, rtype, nprev):
        self.type = rtype
        self.nprev = nprev
        assert self.validType

    def validType(self):
        return self.type in self.types_allowed

    def get_attach_point(self, prev):
        """
        Get the mean attachment point of where the start of the next part
        should be, given the previous ones and their relations

        :param prev: TODO
        :return:
            pos: TODO
        """
        if self.type == 'unihist':
            pos = self.gpos
        else:
            part = prev[self.attach_spot]
            if self.type == 'start':
                subtraj = part.motor[0]
                pos = subtraj[0]
            elif self.type == 'end':
                subtraj = part.motor[-1]
                pos = subtraj[-1]
            elif self.type == 'mid':
                bspline = part.motor_spline[:,:,self.subid_spot]
                pos = bspline_eval[self.eval_spot_token, bspline]
            else:
                raise TypeError('invalid relation type')

        return pos

class RelationIndependent(Relation):
    def __init__(self, rtype, nprev, gpos=None):
        Relation.__init__(self, rtype, nprev)
        self.gpos = gpos

class RelationAttach(Relation):
    def __init__(self, rtype, nprev, attach_spot):
        Relation.__init__(self, rtype, nprev)
        self.attach_spot = attach_spot

class RelationAttachAlong(RelationAttach):
    def __init__(self, rtype, nprev, attach_spot, nsub, subid_spot, ncpt):
        RelationAttach.__init__(self, rtype, nprev, attach_spot)
        self.subid_spot = subid_spot
        self.ncpt = ncpt
        self.nsub = nsub
        self.eval_spot_type = []
        self.eval_spot_token = []
