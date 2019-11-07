from __future__ import division, print_function
import matplotlib.pyplot as plt

from pybpl.library import Library
from pybpl.model import CharacterModel


def main():
    print('generating character...')
    lib = Library(lib_dir='./lib_data', use_hist=True)
    model = CharacterModel(lib)
    fig, axes = plt.subplots(nrows=10, ncols=3, figsize=(1.5, 5))
    for i in range(10):
        for j in range(3):
            img = model.sample_image_sequential()
            axes[i, j].imshow(img, cmap='Greys')
            axes[i, j].tick_params(
                which='both',
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False
            )
        axes[i, 0].set_ylabel('%i' % i, fontsize=10)
    plt.show()


if __name__ == '__main__':
    main()
