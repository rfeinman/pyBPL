class Type(object):

    def __init__(self,k,P,R):
        self.k = k
        self.part_types = P
        self.relation_types = R
        
    def parameters(self):
        '''
        return list of parameters
        '''
        pass

    def train(self):
        '''
        makes params require grad
        '''
        pass
    def eval(self):
        '''
        makes params require no grad
        '''
        pass
    def to(self,device):
        '''
        moves parameters to device
        '''
        pass

class Token(object):

    def __init__(self,P,R):
        self.part_tokens = P
        self.relation_tokens = R

    def parameters(self):
        '''
        return list of parameters
        '''
        pass

    def train(self):
        '''
        makes params require grad
        '''
        pass
    def eval(self):
        '''
        makes params require no grad
        '''
        pass
    def to(self,device):
        '''
        moves parameters to device
        '''
        pass





class TypeDist():
    '''
    prior over Type
    '''
    def __init__(self,lib)
        pass
    def sample_type():
        '''
        Note: Should only be called from Model
        Note: Should return Type object
        '''
        pass
    def score_type(_type):
        '''
        Note: Should only be called from Model
        Note: should return a log probability
        '''
        pass

class TokenDist():
    '''
    prior over Token
    '''
    def __init__(self,lib)
        pass
    def sample_token(self,_type):
        '''
        Note: Should only be called from Model
        Note: Should return Token object
        '''
        pass
    def score_token(self,token):
        '''
        Note: Should only be called from Model
        Note: Should return a log probability
        '''
        pass

class ImageDist():
    '''
    Likelihood Distribution
    '''
    def __init__(self,lib)
        pass

    def sample_image(self,_type):
        '''
        Note: Should only be called from Model
        '''
        pass
    def score_image(self,token):
        '''
        Note: Should only be called from Model
        Note: Should return a log probability
        '''
        pass

class Model(object):
    '''
    Sampling from and Scoring according to the graphical model
    '''
    def __init__(self,lib):
        self.type_dist = TypeDist(lib)
        self.token_dist = TokenDist(lib)
        self.image_dist = ImageDist(lib)

    def sample_type(self):
        return self.type_dist.sample_type()
        
    
    def sample_token(self,_type):
        return self.token_dist.sample_token(_type)

    def sample_image(self,token):
        pass

    def score_type(self,_type):
        return self.type_dist.score_type(_type)

    def score_token(self,_type,token):
        return self.token_dist.score_token(_type,token)

    def score_image(self,token,image):
        return self.image_dist.score_image(token,image)



# Optimization would look something like this

model = Model(lib)
_type = model.sample_type()
token = model.sample_token(_type)

optimizer = torch.optim.Adam([{'params': _type.parameters()},
                              {'params': token.parameters()}], 
                              lr=0.001)

# Set requires_grad to True
_type.train()
token.train()

for idx in iters:
    optimizer.zero_grad()
    type_score = model.score_type(_type)
    token_score = model.score_token(_type,token)
    image_score = model.score_image(token,im)
    score = type_score + token_score + image_score
    loss = -score
    loss.backward()
    optimizer.step()

