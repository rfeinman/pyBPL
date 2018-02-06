#relation stuff

class Relation(object):
	types_allowed = ['unihist','start','end','mid']

	def validType(self):
		return self.rtype in types_allowed


	#self.type = []
	#self.nprev

class RelationIndependent(Relation):
	def __init__(self, rtype, nprev, gpos=[]):
		self.rtype = rtype
		self.nprev = nprev
		self.gpos = gpos
		assert self.validType

class RelationAttach(Relation):
	def __init__(self, rtype, nprev, attach_spot):
		self.rtype = rtype
		self.nprev = nprev
		self.attach_spot = attach_spot
		assert self.validType

class RelationAttachAlong(RelationAttach):
	def __init__(self, rtype, nprev, attach_spot, nsub, subid_spot, ncpt):
		super(RelationAttachAlong,self).__init__(rtype, nprev, attach_spot) #hope this works
		self.subid_spot = subid_spot
		self.ncpt = ncpt
		self.nsub = nsub
		eval_spot_type = []
		eval_spot_token = []



