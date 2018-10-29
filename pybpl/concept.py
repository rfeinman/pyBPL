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



# ------------------------ #
# parent 'Concept' classes
# ------------------------ #

class ConceptToken(object):
    """
    Abstract base class for concept tokens

    Parameters
    ----------
    P : list of PartToken
        TODO
    R : list of RelationToken
        TODO
    """
    __metaclass__ = ABCMeta

    def __init__(self, P, R):
        assert isinstance(P, list)
        assert isinstance(R, list)
        assert len(P) == len(R)
        for ptoken, rtoken in zip(P, R):
            assert isinstance(ptoken, PartToken)
            assert isinstance(rtoken, RelationToken)
        self.P = P
        self.R = R


class Concept(object):
    """
    An abstract base class for concepts. A concept is a probabilistic program
    that samples Concept tokens. Concepts are made up of parts and relations.

    Parameters
    ----------
    k : tensor
        scalar; part count
    P : list of Part
        part type list
    R : list of Relation
        relation type list
    """
    __metaclass__ = ABCMeta

    def __init__(self, k, P, R):
        assert isinstance(P, list)
        assert isinstance(R, list)
        assert len(P) == len(R)
        assert k > 0
        for ptype, rtype in zip(P, R):
            assert isinstance(ptype, Part)
            assert isinstance(rtype, Relation)
        self.k = k
        self.P = P
        self.R = R

    @abstractmethod
    def sample_token(self):
        P = []
        R = []
        for p, r in zip(self.P, self.R):
            # sample part token
            ptoken = p.sample_token()
            # sample relation token
            rtoken = r.sample_token()
            # sample part position from relation token
            ptoken.position = rtoken.sample_location(P)
            # append them to the list
            P.append(ptoken)
            R.append(rtoken)
        token = ConceptToken(P, R)

        return token

    def score_token(self, token):
        ll = 0.
        for i in range(self.k):
            ll = ll + self.P[i].score_token(token.P[i])
            ll = ll + self.R[i].score_token(token.R[i])
            ll = ll + token.R[i].score_location(token.P[i].position, token.P[:i])

        return ll


# ------------------------- #
# child 'Character' classes
# ------------------------- #

class CharacterToken(ConceptToken):
    """
    Character token stores the token-level parameters of a character sample

    Parameters
    ----------
    P : list of StrokeToken
        TODO
    R : list of RelationToken
        TODO
    affine : TODO
        TODO
    epsilon : TODO
        TODO
    blur_sigma : TODO
        TODO
    parameters : TODO
        TODO
    """
    def __init__(self, P, R, affine, epsilon, blur_sigma, parameters):
        super(CharacterToken, self).__init__(P, R)
        for ptoken in P:
            assert isinstance(ptoken, StrokeToken)
        self.affine = affine
        self.epsilon = epsilon
        self.blur_sigma = blur_sigma
        self.parameters = parameters

    @property
    def pimg(self):
        pimg, _ = rendering.apply_render(
            self.P, self.affine, self.epsilon, self.blur_sigma, self.parameters
        )

        return pimg

    @property
    def ink_off_page(self):
        _, ink_off_page = rendering.apply_render(
            self.P, self.affine, self.epsilon, self.blur_sigma, self.parameters
        )

        return ink_off_page

    def sample_image(self):
        binom = dist.binomial.Binomial(1, self.pimg)
        image = binom.sample()

        return image

    def score_image(self, image):
        binom = dist.binomial.Binomial(1, self.pimg)
        ll = binom.log_prob(image)
        ll = torch.sum(ll)

        return ll


class Character(Concept):
    """
    A Character is a probabilistic program that samples Character tokens.
    Parts are strokes, and relations are either [independent, attach,
    attach-along].

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
        TODO

        Returns
        -------
        affine : TODO
            affine transformation
        """
        warnings.warn('skipping affine warp for now.')
        affine = None

        return affine

    def sample_image_noise(self):
        """
        TODO

        Returns
        -------
        epsilon : float
            scalar; image noise quantity
        """
        #epsilon = CPD.sample_image_noise(self.parameters)
        warnings.warn('using fixed image noise for now.')
        # set rendering parameters to minimum noise
        epsilon = self.parameters.min_epsilon

        return epsilon

    def sample_image_blur(self):
        """
        TODO

        Returns
        -------
        blur_sigma: float
            scalar; image blur quantity
        """
        #blur_sigma = CPD.sample_image_blur(self.parameters)
        warnings.warn('using fixed image blur for now.')
        # set rendering parameters to minimum noise
        blur_sigma = self.parameters.min_blur_sigma

        return blur_sigma
