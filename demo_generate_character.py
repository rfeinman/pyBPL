from __future__ import division, print_function
import warnings
import matplotlib.pyplot as plt

from pybpl.library.library import Library
from pybpl.character.character import Character
from pybpl.generate_character import generate_type


def display_type(S, R):
    print('----BEGIN CHARACTER TYPE INFO----')
    print('num strokes: %i' % len(S))
    for i in range(len(S)):
        print('Stroke #%i:' % i)
        print('\tsub-stroke ids: ', list(S[i].ids.numpy()))
        print('\trelation type: %s' % R[i].type)
    print('----END CHARACTER TYPE INFO----')

def main():
    print('generating character...')
    warnings.warn(
        "'generate_character' not yet fully implemented. Generating character "
        "type (template) for now."
    )
    lib = Library(lib_dir='./lib_data')
    # generate the character type
    S, R = generate_type(lib)
    display_type(S, R)
    # initialize the motor program
    char = Character(S, R, lib)
    # sample a few character tokens and visualize them
    plt.figure()
    for i in range(4):
        exemplar = char.sample_token()
        im = exemplar.image.numpy()
        plt.subplot(2,2,i+1)
        plt.imshow(im, cmap='Greys', vmin=0, vmax=1)
        plt.title('Token #%i' % (i+1))
    plt.show()

if __name__ == '__main__':
    main()