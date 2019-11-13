from pybpl.library import Library
from pyprob import Model
from pybpl.model import CharacterModel
from pybpl.concept import CharacterToken
import pyprob
import torch


class BPL(Model):
    def __init__(self, lib_dir='../lib_data'):
        super().__init__(name="BPL")
        self.lib = Library(lib_dir=lib_dir, use_hist=True)
        self.model = CharacterModel(self.lib)

    def forward(self):
        # sample affine warp
        affine = self.model.token_dist.sample_affine()  # (4,) tensor

        # sample rendering parameters
        epsilon = self.model.token_dist.sample_image_noise()
        blur_sigma = self.model.token_dist.sample_image_blur()

        k = self.model.type_dist.sample_k()
        stroke_types = []
        relation_types = []
        stroke_tokens = []
        relation_tokens = []
        partial_image_probss = []
        for stroke_id in range(k):
            stroke_type = self.model.type_dist.stroke_type_dist.sample_stroke_type(k)
            relation_type = \
                self.model.type_dist.relation_type_dist.sample_relation_type(
                    stroke_types)
            stroke_token = \
                self.model.token_dist.stroke_token_dist.sample_stroke_token(
                    stroke_type)
            relation_token = \
                self.model.token_dist.relation_token_dist.sample_relation_token(
                    relation_type)

            # sample part position from relation token
            stroke_token.position = self.model.token_dist.sample_location(
                relation_token, stroke_tokens)

            stroke_types.append(stroke_type)
            relation_types.append(relation_type)
            stroke_tokens.append(stroke_token)
            relation_tokens.append(relation_token)

            # evaluate partial image probs
            partial_character_token = CharacterToken(
                stroke_tokens[-1:], relation_tokens[-1:], affine, epsilon,
                blur_sigma)
            partial_image_probs = self.model.image_dist.get_pimg(
                partial_character_token)
            partial_image_probss.append(partial_image_probs)
            pyprob.tag(partial_image_probs, address='partial_image_{}'.format(
                stroke_id))

        character_token = CharacterToken(stroke_tokens, relation_tokens,
                                         affine, epsilon, blur_sigma)
        image_probs = torch.clamp(sum(partial_image_probss), 0, 1)
        image_dist = pyprob.distributions.Bernoulli(image_probs)
        pyprob.observe(image_dist, name='image')

        return character_token

        # if return_partial_image_probss:
        #     return image_dist.sample(), partial_image_probss
        # else:
        #     return image_dist.sample()


        # char_type = self.model.sample_type()
        # char_token = self.model.sample_token(char_type)

        # image_dist = self.model.image_dist.image_dist(char_token)
        # pyprob.observe(image_dist, name='image')
        # return char_type, char_token
