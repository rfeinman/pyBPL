"""
generate exemplar function
"""
from __future__ import division, print_function
from MotorProgram import MotorProgram
import CPD


def generate_exemplar(template,libclass):
	M = MotorProgram(template)
	print('M.ns', M.ns)
	#sample stroke params
	for i in range(M.ns):
		if M.S[i].R.rtype == 'mid':
			print("M.S[i].R :", M.S[i].R)
			M.S[i].R.eval_spot_token = CPD.sample_relation_token(
				libclass, M.S[i].R.eval_spot_type)

		M.S[i].pos_token = CPD.sample_position(libclass, M.S[i].R, M.S[0:i]) #check that this does what I want, slicewise
		M.S[i].shapes_token = CPD.sample_shape_token(libclass, M.S[i].shapes_type)
		M.S[i].invscales_token = CPD.sample_invscale_token(libclass, M.S[i].invscales_type)


	M.A = CPD.sample_affine(libclass)

	#set rendering params to minimum noise
	M.blur_sigma = template.parameters.min_blur_sigma
	M.epsilon = template.parameters.min_epsilon

	#sample image
	M.I = CPD.sample_image(M.pimg)

	return M, template
