#generate_character 


#import statements or whatever



def generate_character(libclass, ns=None):
	if ns = None:
		ns = CPD.sample_number(libclass)

	template = MotorProgram(ns)
	template.parameters = defaultps() #need to deal with this - dealt with

	for i in range(ns):
		template.S[i].R = CPD.sample_relation_type(libclass,template.S[0:i]) #oh god check this
		template.S[i].ids = CPD.sample_sequence(libclass,ns)
		template.S[i].shapes_type = CPD.sample_shape_type(libclass,template.S[i].ids)
		template.S[i].invscales_type = CPD.sample_invscales_type(libclass,template.S[i].ids)

	return template, lambda: generate_exemplar(template,libclass)
