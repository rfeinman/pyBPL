from __future__ import division, print_function
from abc import ABCMeta,abstractmethod
import warnings
import torch
import torch.distributions

from .library import Library
from .relation import (Relation,RelationIndependent,RelationAttach,
                      RelationAttachAlong)

from .part import Stroke
from .concept import Concept,Character
from .splines import bspline_gen_s

# list of acceptable dtypes for 'k' parameter
int_types = [torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64]



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


lib = Library('../lib_data/')
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

