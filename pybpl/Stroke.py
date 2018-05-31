#stroke

#import statements
import copy

class Stroke(object):
    def __init__(self, args=None):
        # need to verify that this copies and doesn't point
        if isinstance(args, Stroke):
            self.R = copy.deepcopy(args.R) #idk if this is right
            # idk if this is right, assuming scalar
            self.ids = copy.copy(args.ids)
            self.invscales_type = args.invscales_type.clone() #assuming Tensor
            #print 'self.invscales_type:', self.invscales_type
            #print 'args.invscales_type:', args.invscales_type
            self.shapes_type = args.shapes_type.clone() #assuming Tensor
        #other params
        else:
            #type level
            #print 'bad trigger'
            self.R = []
            self.ids = []
            self.invscales_type = []
            self.shapes_type = []

        #token level
        self.pos_token = []
        self.invscales_token = []
        self.shapes_token = []



    @property
    def nsub(self):
        return len(self.ids)

    @property
    def motor(self):
        return rendering.to_motor(
            self.shapes_token,self.invscales_token,self.pos_token
        )


    #can consider computing motor, will skip for now
