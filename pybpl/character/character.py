"""
Motor program class definition
"""
from __future__ import print_function, division
import torch
import torch.distributions as dist

from pybpl.classes.stroke import Stroke, StrokeType
from pybpl.classes.relations import Relation
from pybpl.classes.library import Library
from pybpl.classes import CPD
from pybpl.classes import UtilMP
from pybpl.parameters import defaultps
from pybpl.rendering import render_image
from .concept import Concept, ConceptType

class CharacterType(ConceptType):
    def __init__(self, S, R):
        for stype in S:
            assert isinstance(stype, StrokeType)
        for rtype in R:
            assert isinstance(rtype, Relation)
        assert len(S) == len(R)
        ConceptType.__init__(self)
        # the list of stroke types
        self.S = S
        # the list of relations
        self.R = R

    @property
    def ns(self):
        # get number of strokes
        return len(self.S)

class Character(Concept):

    def __init__(self, ctype, lib):
        """
        Constructor

        :param ctype: [CharacterType]
        :param lib: [Library]
        """
        assert isinstance(ctype, CharacterType)
        assert isinstance(lib, Library)
        Concept.__init__(self)
        self.ctype = ctype
        self.lib = lib
        self.parameters = defaultps()

    @property
    def ns(self):
        # get number of strokes
        return len(self.ctype.S)

    def sample_token(self):
        """
        Sample a token from the motor program

        :return:
            image: [(m,n) tensor] token (image)
        """
        # sample the token-level stroke params
        strokes = []
        for stype, r in zip(self.ctype.S, self.ctype.R):
            # TODO - need to do something about updating eval_spot_type/token?
            if r.type == 'mid':
                r.eval_spot_token = CPD.sample_relation_token(self.lib, r.eval_spot_type)
            pos_token = CPD.sample_position(self.lib, r, strokes)
            shapes_token = CPD.sample_shape_token(self.lib, stype.shapes_type)
            invscales_token = CPD.sample_invscale_token(self.lib, stype.invscales_type)
            s = Stroke(stype, pos_token, shapes_token, invscales_token)
            strokes.append(s)

        # sample affine warp
        affine = CPD.sample_affine(self.lib)

        # set rendering parameters to minimum noise
        blur_sigma = self.parameters.min_blur_sigma
        epsilon = self.parameters.min_epsilon
        # self.blur_sigma = CPD.sample_image_blur(self.parameters)
        # self.epsilon = CPD.sample_image_noise(self.parameters)

        # get probability map of an image
        pimg, _ = self.__apply_render(strokes, affine, epsilon, blur_sigma)
        # sample the image
        image = sample_image(pimg)

        return image

    def has_relations(self, list_sid=None):
        if list_sid is None:
            list_sid = range(self.ns)
        present = [self.ctype.R[sid] is not None for sid in list_sid]
        assert all(present) or not any(present), \
            'error: all relations should be present or not'
        out = all(present)

        return out

    def istied(self, varargin):
        raise NotImplementedError

    def __apply_warp(self, strokes, affine):
        motor_unwarped = [stroke.motor for stroke in strokes]
        if affine is None:
            motor_warped = motor_unwarped
        else:
            raise NotImplementedError(
                "'apply_warp' method not yet implemented."
            )

        return motor_warped

    def __apply_render(self, strokes, affine, epsilon, blur_sigma):
        motor_warped = self.__apply_warp(strokes, affine)
        flat_warped = UtilMP.flatten_substrokes(motor_warped)
        pimg, ink_off_page = render_image(
            flat_warped, epsilon, blur_sigma, self.parameters
        )

        return pimg, ink_off_page


def sample_image(pimg):
    binom = dist.binomial.Binomial(1, pimg)
    image = binom.sample()

    return image

def score_image(image, pimg):
    binom = dist.binomial.Binomial(1, pimg)
    ll = binom.log_prob(image)

    return ll