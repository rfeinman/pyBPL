from __future__ import division, print_function
from abc import ABCMeta, abstractmethod

from ..parameters import defaultps
from ..concept import ConceptType, CharacterType, ConceptToken, CharacterToken


class ConceptTokenDist(object):
    """
    Defines the distribution P(Token | Type) for concepts
    """
    __metaclass__ = ABCMeta

    def __init__(self, lib):
        self.lib = lib
        self.pdist = PartTokenDist(lib)
        self.rdist = RelationTokenDist(lib)

    @abstractmethod
    def sample_location(self, rtoken, prev_parts):
        pass

    @abstractmethod
    def score_location(self, rtoken, prev_parts, loc):
        pass

    @abstractmethod
    def sample_token(self, ctype):
        """
        Parameters
        ----------
        ctype : ConceptType

        Returns
        -------
        ctoken : ConceptToken
        """
        assert isinstance(ctype, ConceptType)
        P = []
        R = []
        for p, r in zip(ctype.part_types, ctype.relation_types):
            # sample part token
            ptoken = self.pdist.sample_part_token(p)
            # sample relation token
            rtoken = self.rdist.sample_relation_token(r)
            # sample part position from relation token
            ptoken.position = self.sample_location(rtoken, P)
            # append them to the list
            P.append(ptoken)
            R.append(rtoken)
        ctoken = ConceptToken(P, R)

        return ctoken

    @abstractmethod
    def score_token(self, ctype, ctoken):
        """
        Parameters
        ----------
        ctype : ConceptType
        ctoken : ConceptToken

        Returns
        -------
        ll : tensor
        """
        ll = 0.
        for i in range(ctype.k):
            ll = ll + self.pdist.score_part_token(
                ctype.part_types[i], ctoken.part_tokens[i]
            )
            ll = ll + self.rdist.score_relation_token(
                ctype.relation_types[i], ctoken.relation_tokens[i]
            )
            ll = ll + self.score_location(
                ctoken.relation_tokens[i], ctoken.part_tokens[:i],
                ctoken.part_tokens[i].position
            )

        return ll


class CharacterTokenDist(ConceptTokenDist):
    """
    Defines the distribution P(Token | Type) for characters
    """
    def __init__(self, lib):
        super(CharacterTokenDist, self).__init__(lib)
        self.pdist = StrokeTokenDist(lib)
        self.default_ps = defaultps()

    def sample_location(self, rtoken, prev_parts):
        pass

    def score_location(self, rtoken, prev_parts, loc):
        pass

    def sample_affine(self):
        """
        Sample an affine warp
        TODO: update this function. right now it returns None

        Returns
        -------
        affine : (4,) tensor
            affine transformation
        """
        # set affine to None for now
        affine = None

        return affine

    def score_affine(self, affine):
        return 0.

    def sample_image_noise(self):
        """
        Sample an "epsilon," i.e. image noise quantity
        TODO: update this function. right now it returns fixed quantity

        Returns
        -------
        epsilon : tensor
            scalar; image noise quantity
        """
        # set rendering parameters to minimum noise for now
        epsilon = self.default_ps.min_epsilon

        return epsilon

    def score_image_noise(self, epsilon):
        return 0.

    def sample_image_blur(self):
        """
        Sample a "blur_sigma," i.e. image blur quantity
        TODO: update this function. right now it returns fixed quantity

        Returns
        -------
        blur_sigma: tensor
            scalar; image blur quantity
        """
        # set rendering parameters to minimum noise for now
        blur_sigma = self.default_ps.min_blur_sigma

        return blur_sigma

    def score_image_blur(self, blur_sigma):
        return 0.

    def sample_token(self, ctype):
        """
        Sample a character token from P(Token | Type = type).
        Note: should only be called from Model

        Parameters
        ----------
        ctype : CharacterType
            TODO

        Returns
        -------
        ctoken : CharacterToken
            character token
        """
        # sample part and relation tokens
        concept_token = super(CharacterTokenDist, self).sample_token()

        # sample affine warp
        affine = self.sample_affine() # (4,) tensor

        # sample rendering parameters
        epsilon = self.sample_image_noise()
        blur_sigma = self.sample_image_blur()

        # create the character token
        ctoken = CharacterToken(
            concept_token.part_tokens, concept_token.relation_tokens, affine,
            epsilon, blur_sigma
        )

        return ctoken

    def score_token(self, ctype, ctoken):
        """
        Compute the log-probability of a concept token,
        log P(Token = token | Type = type).
        Note: Should only be called from Model

        Parameters
        ----------
        ctype : CharacterType
            concept type to condition on
        ctoken : CharacterToken
            concept token to be scored

        Returns
        -------
        ll : tensor
            scalar; log-likelihood of the token
        """
        ll = super(CharacterTokenDist, self).score_token(ctype, ctoken)
        ll += self.score_affine(ctoken.affine)
        ll += self.score_image_noise(ctoken.epsilon)
        ll += self.score_image_blur(ctoken.blur_sigma)

        return ll


class PartTokenDist(object):
    __metaclass__ = ABCMeta
    def __init__(self, lib):
        self.lib = lib

    @abstractmethod
    def sample_part_token(self, ptype):
        pass

    @abstractmethod
    def score_part_token(self, ptype, ptoken):
        pass


class StrokeTokenDist(PartTokenDist):
    def __init__(self, lib):
        super(PartTokenDist, self).__init__(lib)

    def sample_part_token(self, ptype):
        """
        TODO
        """

    def score_part_token(self, ptype, ptoken):
        """
        TODO
        """


class RelationTokenDist(object):
    __metaclass__ = ABCMeta

    def __init__(self, lib):
        self.lib = lib

    def sample_relation_token(self, rtype):
        """
        TODO
        """
        pass

    def score_relation_token(self, rtype, rtoken):
        """
        TODO
        """
        pass
