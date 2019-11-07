from __future__ import division, print_function
import torch

from .type_dist import CharacterTypeDist
from .token_dist import CharacterTokenDist
from .image_dist import CharacterImageDist
from ..concept import CharacterToken


class CharacterModel(object):
    """
    Sampling from and Scoring according to the graphical model. The model is
    defined as P(Type, Token, Image) = P(Type)*P(Token | Type)*P(Image | Token).
    The 3 component distributions P(Type), P(Token | Type), and P(Image | Token)
    are denoted 'type_dist', 'token_dist' and 'image_dist', respectively.
    """
    def __init__(self, lib):
        self.type_dist = CharacterTypeDist(lib)
        self.token_dist = CharacterTokenDist(lib)
        self.image_dist = CharacterImageDist(lib)

    def sample_type(self, k=None):
        return self.type_dist.sample_type(k)

    def score_type(self, ctype):
        return self.type_dist.score_type(ctype)

    def sample_token(self, ctype):
        return self.token_dist.sample_token(ctype)

    def score_token(self, ctype, ctoken):
        return self.token_dist.score_token(ctype, ctoken)

    def sample_image(self, ctoken):
        return self.image_dist.sample_image(ctoken)

    def score_image(self, ctoken, image):
        return self.image_dist.score_image(ctoken, image)

    def get_pimg(self, ctoken):
        return self.image_dist.get_pimg(ctoken)

    def sample_image_sequential(self, return_partial_image_probss=False):
        # sample affine warp
        affine = self.token_dist.sample_affine()  # (4,) tensor

        # sample rendering parameters
        epsilon = self.token_dist.sample_image_noise()
        blur_sigma = self.token_dist.sample_image_blur()

        k = self.type_dist.sample_k()
        stroke_types = []
        relation_types = []
        stroke_tokens = []
        relation_tokens = []
        partial_image_probss = []
        for stroke_id in range(k):
            stroke_type = self.type_dist.stroke_type_dist.sample_stroke_type(k)
            relation_type = \
                self.type_dist.relation_type_dist.sample_relation_type(
                    stroke_types)
            stroke_token = \
                self.token_dist.stroke_token_dist.sample_stroke_token(
                    stroke_type)
            relation_token = \
                self.token_dist.relation_token_dist.sample_relation_token(
                    relation_type)

            # sample part position from relation token
            stroke_token.position = self.token_dist.sample_location(
                relation_token, stroke_tokens)

            stroke_types.append(stroke_type)
            relation_types.append(relation_type)
            stroke_tokens.append(stroke_token)
            relation_tokens.append(relation_token)

            # evaluate partial image probs
            partial_character_token = CharacterToken(
                stroke_tokens[-1:], relation_tokens[-1:], affine, epsilon,
                blur_sigma)
            partial_image_probs = self.image_dist.get_pimg(
                partial_character_token)
            partial_image_probss.append(partial_image_probs)

        image_probs = torch.clamp(sum(partial_image_probss), 0, 1)
        image_dist = torch.distributions.Bernoulli(image_probs)
        if return_partial_image_probss:
            return image_dist.sample(), partial_image_probss
        else:
            return image_dist.sample()


def fit_image(im, lib):
    # Optimization would look something like this

    model = CharacterModel(lib)
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
