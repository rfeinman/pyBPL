#import statments 
import copy

#motor program 


class MotorProgram(object): #do i need to inherit from object??? sure, makes things easier
	def __init__(self, *args):
		#set variables:
		self.I = []
		self.S = []
		self.parameters = []


		if isinstance(args, int):
			ns = args
			for i in range(ns):
				self.S[i] = Stroke()
		elif isinstance(args, MotorProgram):
			Template = args
			for i in range(Template.ns):
				self.S[i] = Stroke(Template.S[i])
			self.parameters = copy.copy(Template.parameters) #this might break if mcmc comes online
		else: 
			error('invalid constructor')


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

	#[pimg, ink_off_page] = apply_render(this) -- this is the thing that needs to be differentiable

	def apply_render(self):
		#apply the render, also skip the "motor" part
		return pimg, ink_off_page

	@property
	def pimg(self):
		return apply_render(self)[0] #hopefully this will work

	#this is probably not the correct way to do this. May mess with rendering.
	@property
	def ink_off_page(self):
		return apply_render(self)[1]



