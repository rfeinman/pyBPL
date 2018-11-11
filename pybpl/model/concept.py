"""
Concepts are probabilistic programs that sample concept tokens. A concept
contains a sequence of parts and a sequence of relations to connect each
part to previous parts. The Concept class is an abstract class, and must be
inherited from by derivative classes. It defines the general overarching
structure that child classes need to adhere to.
"""
from __future__ import division, print_function
from abc import ABCMeta, abstractmethod
import warnings
import torch
import torch.distributions as dist

from . import rendering
from .parameters import defaultps
from .library import Library
from .part import Part, Stroke, PartToken, StrokeToken
from .relation import Relation, RelationToken


class Character(Concept):
    """
    A Character (type) is a probabilistic program that samples Character tokens.
    Parts are strokes, and relations are either [independent, attach,
    attach-along]. 'Character' defines the distribution P(Token | Type = type).

    Parameters
    ----------
    k : tensor
        scalar; part count
    P : list of Stroke
        part type list
    R : list of Relation
        relation type list
    lib : Library
        library instance
    """

    def __init__(self, k, P, R, lib):
        super(Character, self).__init__(k, P, R)
        for p in P:
            assert isinstance(p, Stroke)
        assert isinstance(lib, Library)
        self.lib = lib
        self.parameters = defaultps()

    def optimizable_parameters(self, eps=1e-4):
        """
        Returns a list of parameters that can be optimized via gradient descent.
        Includes lists of lower and upper bounds, with one per parameter.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        params : list
            optimizable parameters
        lbs : list
            lower bound for each parameter
        ubs : list
            upper bound for each parameter
        """
        params = []
        lbs = []
        ubs = []
        for p, r in zip(self.P, self.R):
            params_p, lbs_p, ubs_p = p.optimizable_parameters(eps=eps)
            params.extend(params_p)
            lbs.extend(lbs_p)
            ubs.extend(ubs_p)
            params_r, lbs_r, ubs_r = r.optimizable_parameters(eps=eps)
            params.extend(params_r)
            lbs.extend(lbs_r)
            ubs.extend(ubs_r)

        return params, lbs, ubs

    def sample_token(self):
        """
        Sample a character token

        Returns
        -------
        token : CharacterToken
            character token
        """
        # sample part and relation tokens
        concept_token = super(Character, self).sample_token()

        # sample affine warp
        affine = self.sample_affine() # (4,) tensor

        # sample rendering parameters
        epsilon = self.sample_image_noise()
        blur_sigma = self.sample_image_blur()

        # create the character token
        token = CharacterToken(
            concept_token.P, concept_token.R, affine, epsilon, blur_sigma,
            self.parameters
        )

        return token

    def sample_affine(self):
        """
        Sample an affine warp

        Returns
        -------
        affine : (4,) tensor
            affine transformation
        """
        warnings.warn('skipping affine warp for now.')
        affine = None

        return affine

    def sample_image_noise(self):
        """
        Sample an "epsilon," i.e. image noise quantity

        Returns
        -------
        epsilon : tensor
            scalar; image noise quantity
        """
        warnings.warn('using fixed image noise for now.')
        # set rendering parameters to minimum noise
        epsilon = self.parameters.min_epsilon

        return epsilon

    def sample_image_blur(self):
        """
        Sample a "blur_sigma," i.e. image blur quantity

        Returns
        -------
        blur_sigma: tensor
            scalar; image blur quantity
        """
        warnings.warn('using fixed image blur for now.')
        # set rendering parameters to minimum noise
        blur_sigma = self.parameters.min_blur_sigma

        return blur_sigma
