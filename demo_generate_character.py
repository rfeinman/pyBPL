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
    lib = Library(lib_dir='./lib_data')
    S, R = generate_type(lib)
    print('strokes: ', S)
    print('relations: ', R)
    # char = Character(S, R, lib)
    # exemplar = char.sample_token()

if __name__ == '__main__':
    main()