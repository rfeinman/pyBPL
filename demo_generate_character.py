from __future__ import division, print_function
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
    fig, axes = plt.subplots(nrows=10, ncols=3, figsize=(1.5, 5))
    for i in range(10):
        S, R = type_dist.sample_type()
        ll = type_dist.score_type(S, R)
        print('type %i' % i)
        display_type(S, R)
        print('log-likelihood: %0.2f \n' % ll.item())
        # initialize the motor program
        char = Character(S, R, lib)
        # sample a few character tokens and visualize them
        for j in range(3):
            _, exemplar = char.sample_token()
            im = exemplar.numpy()
            axes[i,j].imshow(im, cmap='Greys', vmin=0, vmax=1)
            axes[i,j].tick_params(
                which='both',
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False
            )
        axes[i,0].set_ylabel('%i' % i, fontsize=10)
    plt.show()

if __name__ == '__main__':
    main()