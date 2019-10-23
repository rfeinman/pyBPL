import data
import model
import pyprob
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def get_color(k):
    """Color map for the stroke of index k"""

    scol = ['r', 'g', 'b', 'm', 'c']
    ncol = len(scol)
    if k < ncol:
        out = scol[k]
    else:
        out = scol[-1]
    return out


def plot_parse(ax, char_token):
    for i, stroke_token in enumerate(char_token.part_tokens):
        color = get_color(i)
        for j, xy in enumerate(stroke_token.motor):
            if j == 0:
                ax.scatter(xy[0, 0], -xy[0, 1], s=20, color=color, marker='o')
            ax.plot(xy[:, 0], -xy[:, 1], color=color, linewidth=2)
    return ax


def get_test_image_ij():
    for i in range(3):
        for j in range(3):
            yield i, j


def get_parse_ij():
    for i in range(3):
        for j in range(3):
            yield i, j + 3


def main():
    bpl = model.BPL()
    bpl.load_inference_network('save/bpl_inference_network')
    omniglot_test_dataset = data.OmniglotDataset(data.test_img_dir,
                                                 data.test_motor_dir)

    num_test_images = 9
    test_images = omniglot_test_dataset[0][:num_test_images]

    num_is_samples = 10
    posteriors = []
    for test_image in test_images:
        posterior = bpl.posterior_results(
            num_traces=num_is_samples,
            inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK,
            observe={'image': test_image}
        )
        posteriors.append(posterior)
        print('Effective sample size = {}'.format(posterior.effective_sample_size.item()))

    # Plotting
    fig, axss = plt.subplots(3, 6, figsize=(6 * 2, 3 * 2))

    for test_image, (i, j) in zip(test_images, get_test_image_ij()):
        axss[i, j].imshow(1 - test_image, cmap='gray')

    for posterior, (i, j) in zip(posteriors, get_parse_ij()):
        char_type_mode, char_token_mode = posterior.mode
        plot_parse(axss[i, j], char_token_mode)

    for axs in axss:
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim(105, 0)
            ax.set_xlim(0, 105)
    axss[0][1].set_title('Images')
    axss[0][4].set_title('Parses')

    fig.legend([Line2D([0], [0], color=get_color(i), lw=2)
                for i in range(5)],
               range(1, 6),
               ncol=5, loc=(0.6, -0.01), frameon=False)
    fig.tight_layout(pad=0)

    filename = 'plots/parse.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


if __name__ == '__main__':
    main()
