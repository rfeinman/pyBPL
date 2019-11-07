from __future__ import division, print_function
import matplotlib.pyplot as plt

from pybpl.library import Library
from pybpl.model import CharacterModel


def main():
    print('generating character...')
    lib = Library(lib_dir='./lib_data', use_hist=True)
    model = CharacterModel(lib)
    fig, axss = plt.subplots(nrows=10, ncols=11, figsize=(5.5, 5), sharex=True,
                             sharey=True)

    for img_id in range(10):
        img, partial_image_probss = model.sample_image_sequential(
            return_partial_image_probss=True)
        for partial_image_id, partial_image_probs in enumerate(partial_image_probss):
            axss[img_id, partial_image_id].imshow(partial_image_probs, cmap='Greys')
        axss[img_id, len(partial_image_probss)].imshow(img, cmap='Greys')
        for i in range(len(partial_image_probss) + 1, 11):
            axss[img_id, i].axis('off')
    for axs in axss:
        for ax in axs:
            ax.tick_params(
                which='both',
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False
            )
    fig.tight_layout(pad=0)
    plt.show()


if __name__ == '__main__':
    main()
