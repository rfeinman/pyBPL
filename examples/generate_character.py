import matplotlib.pyplot as plt

from pybpl.library import Library
from pybpl.model import CharacterModel


def display_type(c):
    print('----BEGIN CHARACTER TYPE INFO----')
    print('num strokes: %i' % c.k)
    for i in range(c.k):
        print('Stroke #%i:' % i)
        print('\tsub-stroke ids: ', list(c.part_types[i].ids.numpy()))
        print('\trelation category: %s' % c.relation_types[i].category)
    print('----END CHARACTER TYPE INFO----')

def main():
    print('generating character...')
    lib = Library(use_hist=True)
    model = CharacterModel(lib)
    fig, axes = plt.subplots(nrows=10, ncols=3, figsize=(1.5, 5))
    for i in range(10):
        ctype = model.sample_type()
        ll = model.score_type(ctype)
        print('type %i' % i)
        display_type(ctype)
        print('log-likelihood: %0.2f \n' % ll.item())
        # sample a few character tokens and visualize them
        for j in range(3):
            ctoken = model.sample_token(ctype)
            img = model.sample_image(ctoken)
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