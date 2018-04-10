#generate_character 

from __future__ import print_function, division
import torch
import CPD
from MotorProgram import MotorProgram
from parameters import defaultps
from generate_exemplar import generate_exemplar
import loadlib as lb


def generate_character(libclass, ns=None):
	if ns is None:
		numstrokes = CPD.sample_number(libclass)
		ns = numstrokes.data[0]
	template = MotorProgram(ns)
	template.parameters = defaultps() #need to deal with this - dealt with
	print('ns: %i' % ns)
	for i in range(ns):
		template.S[i].R = CPD.sample_relation_type(libclass,template.S[0:i]) #oh god check this
		template.S[i].ids = CPD.sample_sequence(libclass,ns)
		template.S[i].shapes_type = CPD.sample_shape_type(libclass,template.S[i].ids)
		template.S[i].invscales_type = CPD.sample_invscales_type(libclass,template.S[i].ids)
	return template, lambda: generate_exemplar(template,libclass)


def main():
	lib = lb.loadlib()
	x,y = generate_character(lib)
	print('generating exemplar:')
	character = y()


if __name__ == '__main__':
	main()