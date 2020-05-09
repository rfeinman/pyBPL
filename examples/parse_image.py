import math
import imageio
import numpy as np
import matplotlib.pylab as plt

from pybpl.util import dist_along_traj
from pybpl.bottomup import generate_random_parses



def plot_stroke(ax, stk, color, lw=2):
    if len(stk) > 1 and dist_along_traj(stk) > 0.01:
        ax.plot(stk[:,0], -stk[:,1], color=color, linewidth=lw)
    else:
        ax.plot(stk[0,0], -stk[0,1], color=color, linewidth=lw, marker='.')

def plot_parse(ax, strokes, lw=2):
    ns = len(strokes)
    colors = ['r','g','b','m','c']
    for i in range(ns):
        plot_stroke(ax, strokes[i], colors[i], lw)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0,105)
    ax.set_ylim(105,0)

def main():
    # load image to numpy binary array
    img = imageio.imread('./image_H.jpg')
    img = np.array(img > 200)

    # generate random parses
    parses = generate_random_parses(img, seed=3)

    # plot parsing results
    nparse = len(parses)
    n = math.ceil(nparse/10)
    m = 10
    fig, axes = plt.subplots(n,m+1,figsize=(m+1, n))
    # first column
    axes[0,0].imshow(img, cmap=plt.cm.binary)
    axes[0,0].set_xticks([]); axes[0,0].set_yticks([])
    axes[0,0].set_title('Input')
    for i in range(1,n):
        axes[i,0].set_axis_off()
    # remaining_columns
    for i in range(n):
        for j in range(1,m+1):
            ix = i*m + (j-1)
            if ix >= nparse:
                axes[i,j].set_axis_off()
                continue
            plot_parse(axes[i,j], parses[ix])
    plt.subplots_adjust(hspace=0., wspace=0.)
    plt.show()


if __name__ == '__main__':
    main()