from __future__ import division, print_function
import warnings
import matplotlib.pyplot as plt

from pybpl.library import Library
from pybpl.concept import Character
from pybpl.ctd import CharacterTypeDist


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
    lib = Library(lib_dir='./lib_data')
    # generate the character type
    type_dist = CharacterTypeDist(lib)
    S, R = type_dist.sample_type()
    display_type(S, R)
    # initialize the motor program
    char = Character(S, R, lib)
    # sample a few character tokens and visualize them
    plt.figure()
    for i in range(4):
        _, exemplar = char.sample_token()
        im = exemplar.numpy()
        plt.subplot(2,2,i+1)
        plt.imshow(im, cmap='Greys', vmin=0, vmax=1)
        plt.title('Token #%i' % (i+1))
    plt.show()

if __name__ == '__main__':
    main()