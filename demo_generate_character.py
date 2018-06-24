from __future__ import division, print_function
import warnings

from pybpl.library.library import Library
from pybpl.character.character import Character
from pybpl.generate_character import generate_type


def main():
    print('generating character...')
    warnings.warn(
        "'generate_character' not yet fully implemented. Generating character "
        "type (template) for now."
    )
    lib = Library(lib_dir='./library')
    ctype = generate_type(lib)
    print('Character type: ', ctype)
    # char = Character(ctype, lib)
    # exemplar = char.sample_token()

if __name__ == '__main__':
    main()