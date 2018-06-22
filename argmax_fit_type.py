from __future__ import division, print_function

import torch
from torch import distributions

from pybpl.classes import MotorProgram
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device("cpu")

projected_grad_ascent = True # use projected gradient ascent for constrained optimization
ns = 2 # number of strokes
k = 3 # number of sub-strokes per stroke
ncpt = 5 # number of control points
lr = 0.01 # learning rate
eps = 0.0001 # tol. for constrained optimization

mvn = distributions.multivariate_normal.MultivariateNormal(torch.zeros(ncpt*2), torch.eye(ncpt*2))
# gamma = distributions.gamma.Gamma(9.,2.)
gamma = distributions.gamma.Gamma(1.,1.)

def test_make_MP():
	# create an synthetic motor program
	# return: M
	M = MotorProgram(ns)
	for sid in range(ns):
		M.S[sid].shapes_type = torch.zeros(ncpt, 2, k, device=device, dtype=dtype)
		M.S[sid].invscales_type = torch.zeros(k, device=device, dtype=dtype)
		for bid in range(k):
			M.S[sid].shapes_type[:,:,bid] = mvn.sample().view(ncpt,2)
			M.S[sid].invscales_type[bid] = gamma.sample()
	return M

def get_variables_MP(M):
	# indicate variables for optimization (requires_grad_)
	# Returns
	#  parameters : [n x 1] list of pointers to parameters
	#  lbs : [n x 1] list of lower bounds (each elem. is a tensor same size as param; empty list indicates no lb)
	#  ubs : [n x 1] list of upper bounds (each elem. is a tensor; empty list indicates no ub)
	parameters = []
	lbs = []
	ubs = []
	for sid in range(ns):

		# shape
		M.S[sid].shapes_type.requires_grad_()
		parameters.append(M.S[sid].shapes_type)
		lbs.append([])
		ubs.append([])

		# scale
		M.S[sid].invscales_type.requires_grad_()
		parameters.append(M.S[sid].invscales_type)
		lbs.append(torch.full(M.S[sid].invscales_type.shape, eps))
		ubs.append([])

	return parameters, lbs, ubs

def obj_fun(M):
	ll = torch.tensor(0., device=device, dtype=dtype) # don't declare requires_grad
	
	# prior on programs
	for sid in range(ns):
		for bid in range(k): # must use inline addition
			ll.add_(mvn.log_prob(M.S[sid].shapes_type[:,:,bid].view(-1)))
			ll.add_(gamma.log_prob(M.S[sid].invscales_type[bid]))

	# likelihood of image
	## not implemented
	
	return ll

if __name__ == '__main__':
	
	M = test_make_MP()
	parameters,lbs,ubs = get_variables_MP(M)
	score_list = []

	for idx in range(1000):
		score = obj_fun(M)
		score.backward()
		score_list.append(score)
		with torch.no_grad():
			for ip,param in enumerate(parameters):				
				param.add_(lr * param.grad) # manual grad. ascent

				if projected_grad_ascent:
					lb = lbs[ip]
					ub = ubs[ip]
					if len(lb)>0:
						torch.max(param, lb, out=param)
					if len(ub)>0:
						torch.min(param, ub, out=param)

				param.grad.zero_()

	# for param in parameters:
	# 	print(param)

	plt.plot(score_list)
	plt.ylabel('log-likelihood')
	plt.xlabel('test')
	plt.show()