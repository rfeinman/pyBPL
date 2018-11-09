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
    Abstract base class for concept tokens. Concept tokens consist of a list
    of PartTokens and a list of RelationTokens. 'ConceptToken' defines the
    distribution P(Image | Token = token)

    Parameters
    ----------
    P : list of PartToken
        part tokens
    R : list of RelationToken
        relation tokens
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

    @abstractmethod
    def optimizable_parameters(self, eps=1e-4):
        pass

    @abstractmethod
    def sample_image(self):
        pass

    @abstractmethod
    def score_image(self, image):
        pass


class Concept(object):
    """
    An abstract base class for concept types. A concept (type) is a
    probabilistic program that samples concept tokens. Concepts are made up of
    parts and relations. 'Concept' defines the distribution
    P(Token | Type = type)

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
    def optimizable_parameters(self, eps=1e-4):
        pass

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
        """
        Compute the log-likelihood of a concept token

        Parameters
        ----------
        token : ConceptToken
            concept token to be scored

        Returns
        -------
        ll : tensor
            scalar; log-likelihood of the token
        """
        ll = 0.
        for i in range(self.k):
            ll = ll + self.P[i].score_token(token.P[i])
            ll = ll + self.R[i].score_token(token.R[i])
            ll = ll + token.R[i].score_location(
                token.P[i].position, token.P[:i]
            )

        return ll


# ------------------------- #
# child 'Character' classes
# ------------------------- #

class CharacterToken(ConceptToken):
    """
    Character tokens hold all token-level parameters of the character. They
    consist of a list of PartTokens and a list of RelationTokens.
    'CharacterToken' defines the distribution P(Image | Token = token)

    Parameters
    ----------
    P : list of StrokeToken
        stroke tokens
    R : list of RelationToken
        relation tokens
    affine : (4,) tensor
        affine transformation
    epsilon : tensor
        scalar; image noise quantity
    blur_sigma : tensor
        scalar; image blur quantity
    parameters : defaultps
        default BPL parameters; will be used for stroke rendering
    """
    def __init__(self, P, R, affine, epsilon, blur_sigma, parameters):
        super(CharacterToken, self).__init__(P, R)
        for ptoken in P:
            assert isinstance(ptoken, StrokeToken)
        self.affine = affine
        self.epsilon = epsilon
        self.blur_sigma = blur_sigma
        self.parameters = parameters

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

    @property
    def pimg(self):
        """
        The binary image probability map. Returns a (H,W) tensor with a value
        in range [0,1] at each entry
        """
        pimg, _ = rendering.apply_render(
            self.P, self.affine, self.epsilon, self.blur_sigma, self.parameters
        )

        return pimg

    @property
    def ink_off_page(self):
        """
        Boolean indicating whether or not there was ink drawn outside of the
        image window
        """
        _, ink_off_page = rendering.apply_render(
            self.P, self.affine, self.epsilon, self.blur_sigma, self.parameters
        )

        return ink_off_page

    def sample_image(self):
        """
        Samples a binary image

        Returns
        -------
        image : (H,W) tensor
            binary image sample
        """
        binom = dist.binomial.Binomial(1, self.pimg)
        image = binom.sample()

        return image

    def score_image(self, image):
        """
        Compute the log-likelihood of a binary image

        Parameters
        ----------
        image : (H,W) tensor
            binary image to score

        Returns
        -------
        ll : tensor
            scalar; log-likelihood of the image
        """
        binom = dist.binomial.Binomial(1, self.pimg)
        ll = binom.log_prob(image)
        ll = torch.sum(ll)

        return ll


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
