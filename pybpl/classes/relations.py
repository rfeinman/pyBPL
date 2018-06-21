"""
Relation class definitions
"""
from __future__ import print_function, division

from pybpl.rendering import bspline_eval


class Relation(object):
    types_allowed = ['unihist', 'start', 'end', 'mid']

    def __init__(self, type, nprev):
        self.type = type
        self.nprev = nprev
        assert self.validType

    def validType(self):
        return self.type in self.types_allowed

class RelationIndependent(Relation):
    def __init__(self, type, nprev, gpos=None):
        Relation.__init__(self, type, nprev)
        self.gpos = gpos

class RelationAttach(Relation):
    def __init__(self, type, nprev, attach_spot):
        Relation.__init__(self, type, nprev)
        self.attach_spot = attach_spot

class RelationAttachAlong(RelationAttach):
    def __init__(self, type, nprev, attach_spot, nsub, subid_spot, ncpt):
        RelationAttach.__init__(self, type, nprev, attach_spot)
        self.subid_spot = subid_spot
        self.ncpt = ncpt
        self.nsub = nsub
        self.eval_spot_type = []
        self.eval_spot_token = []

def get_attach_point(R, prev_strokes):
    """
    Get the mean attachment point of where the start of the next stroke should
    be, given the previous ones and their relations

    :param R: TODO
    :param prev_strokes: TODO
    :return:
        pos: TODO
    """
    if R.type == 'unihist':
        pos = R.gpos
    elif R.type == 'start':
        subtraj = prev_strokes[R.attach_spot].motor[0]
        pos = subtraj[0]
    elif R.type == 'end':
        subtraj = prev_strokes[R.attach_spot].motor[-1]
        pos = subtraj[-1]
    elif R.type == 'mid':
        bspline = prev_strokes[R.attach_spot].motor_spline[:,:,R.subid_spot]
        pos = bspline_eval[R.eval_spot_token, bspline]
    else:
        raise TypeError('invalid relation')

    return pos
