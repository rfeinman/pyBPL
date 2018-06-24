from __future__ import division, print_function
import torch
from pybpl import CPD
from pybpl.character.character import CharacterType, Character


# list of acceptable dtypes for 'ns' parameter
int_types = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]

def generate_type(lib, ns=None):
    """
    Generate a character type by sampling from the prior.

    :param lib: [Library] library class instance
    :param ns: [int or tensor] number of strokes for the character
    :return:
        ctype: [CharacterType] character type
    """

    if ns is None:
        # sample the number of strokes 'ns'
        ns = CPD.sample_number(lib)
    else:
        # make sure 'ns' is correct dtype
        isint = isinstance(ns, int)
        istensorint = isinstance(ns, torch.Tensor) and \
                      ns.shape == torch.Size([]) and \
                      ns.dtype in int_types
        assert isint or istensorint

    # initialize stroke type and relation type lists
    S = []
    R = []
    # for each stroke, sample stroke parameters
    for _ in range(ns):
        # sample the stroke type
        stype = CPD.sample_stroke_type(lib, ns)
        # sample the relation of this stroke to previous strokes
        r = CPD.sample_relation_type(lib, S)
        # append stroke type and relation to their respective lists
        S.append(stype)
        R.append(r)
    # initialize the character type
    ctype = CharacterType(S, R)

    return ctype

def generate_token(lib, ns=None):
    """
    Generate a character token by sampling from the prior. First, sample a
    character type from the prior. Then, sample a token of that type from
    the prior.

    :param lib:
    :param ns:
    :return:
    """
    ctype = generate_type(lib, ns)
    char = Character(ctype, lib)
    exemplar = char.sample_token()

    return exemplar

def generate_program(lib, ns=None):
    """
    Generate a character by sampling from the prior. A character is a motor
    program that can sample new character tokens.

    :param lib:
    :param ns:
    :return:
    """
    ctype = generate_type(lib, ns)
    char = Character(ctype, lib)

    return char
