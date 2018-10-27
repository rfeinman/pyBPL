from __future__ import division, print_function
import matplotlib.pyplot as plt

from pybpl.library import Library
from pybpl.ctd import CharacterTypeDist
from pybpl.concept import Image


def display_type(c):
    print('----BEGIN CHARACTER TYPE INFO----')
    print('num strokes: %i' % c.k)
    for i in range(c.k):
        print('Stroke #%i:' % i)
        print('\tsub-stroke ids: ', list(c.P[i].ids.numpy()))
        print('\trelation category: %s' % c.R[i].category)
    print('----END CHARACTER TYPE INFO----')

def main():
    print('generating character...')
    lib = Library(lib_dir='./lib_data')
    # generate the character type. This is a motor program for generating
    # character tokens
    type_dist = CharacterTypeDist(lib)
    fig, axes = plt.subplots(nrows=10, ncols=3, figsize=(1.5, 5))
    for i in range(10):
        c = type_dist.sample_type()
        ll = type_dist.score_type(c)
        print('type %i' % i)
        display_type(c)
        print('log-likelihood: %0.2f \n' % ll.item())
        # sample a few character tokens and visualize them
        for j in range(3):
            token = c.sample_token()
            img = Image(token).sample_image()
            axes[i,j].imshow(img, cmap='Greys')
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