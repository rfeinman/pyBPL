from __future__ import division, print_function
import torch
from .concept import Character
from .ctd import CharacterTypeDist


# list of acceptable dtypes for 'ns' parameter
int_types = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]

def generate_type(lib, ns=None):
    type_dist = CharacterTypeDist(lib)
    S, R = type_dist.sample_type(k=ns)

    return S, R

def generate_token(lib, ns=None):
    """
    Generate a character token by sampling from the prior. First, sample a
    character type from the prior. Then, sample a token of that type from
    the prior.

    :param lib:
    :param ns:
    :return:
    """
    S, R = generate_type(lib, ns)
    char = Character(S, R, lib)
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
    S, R = generate_type(lib, ns)
    char = Character(S, R, lib)

    return char
