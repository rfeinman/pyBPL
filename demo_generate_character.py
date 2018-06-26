from __future__ import division, print_function
import warnings

from pybpl.library.library import Library
from pybpl.character.character import Character
from pybpl.generate_character import generate_type


def display_type(S, R):
    print('----CHARACTER TYPE INFO----')
    print('num strokes: %i' % len(S))
    for i in range(len(S)):
        print('Stroke #%i:' % i)
        print('\tsub-stroke ids: ', list(S[i].ids.numpy()))
        print('\trelation type: %s' % R[i].type)

def main():
    print('generating character...')
    warnings.warn(
        "'generate_character' not yet fully implemented. Generating character "
        "type (template) for now."
    )
    lib = Library(lib_dir='./lib_data')
    S, R = generate_type(lib)
    display_type(S, R)
    # char = Character(S, R, lib)
    # exemplar = char.sample_token()

if __name__ == '__main__':
    main()