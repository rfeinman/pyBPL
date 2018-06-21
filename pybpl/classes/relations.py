#relation stuff
from __future__ import print_function, division

class Relation(object):
    types_allowed = ['unihist','start','end','mid']

    def validType(self):
        return self.rtype in types_allowed


    #self.type = []
    #self.nprev

class RelationIndependent(Relation):
    def __init__(self, rtype, nprev, gpos=[]):
        self.type = rtype
        self.nprev = nprev
        self.gpos = gpos
        assert self.validType

class RelationAttach(Relation):
    def __init__(self, rtype, nprev, attach_spot):
        self.type = rtype
        self.nprev = nprev
        self.attach_spot = attach_spot
        assert self.validType

class RelationAttachAlong(RelationAttach):
    def __init__(self, rtype, nprev, attach_spot, nsub, subid_spot, ncpt):
        # hope this works
        super(RelationAttachAlong,self).__init__(rtype, nprev, attach_spot)
        self.subid_spot = subid_spot
        self.ncpt = ncpt
        self.nsub = nsub
        eval_spot_type = []
        eval_spot_token = []

def get_attach_point(R, prev_strokes):
    """
    Get the mean attachment point of where the start of the next stroke should
    be, given the previous ones and their relations

    :param R: TODO
    :param prev_strokes: TODO
    :return:
        pos: TODO
    """
    pos = None

    return pos
