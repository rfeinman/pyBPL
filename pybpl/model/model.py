from __future__ import division, print_function
import torch

from .type_dist import TypeDist
from .token_dist import TokenDist
from .image_dist import ImageDist


class Model(object):
    """
    Sampling from and Scoring according to the graphical model
    """
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


def fit_image(im, lib):
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

    for idx in range(100):
        optimizer.zero_grad()
        type_score = model.score_type(_type)
        token_score = model.score_token(_type,token)
        image_score = model.score_image(token,im)
        score = type_score + token_score + image_score
        loss = -score
        loss.backward()
        optimizer.step()

