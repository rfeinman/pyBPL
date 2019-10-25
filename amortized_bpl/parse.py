import data
import model
import pyprob
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch


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


def main(args):
    if args.cuda:
        pyprob.set_device('cuda')
        save_path_suffix = '_cuda'
    else:
        save_path_suffix = ''
    if args.small_lib:
        lib_dir = '../lib_data250'
        save_path_suffix = '{}_250'.format(save_path_suffix)
    else:
        lib_dir = '../lib_data'
        save_path_suffix = '{}'.format(save_path_suffix)
    save_path_suffix = '{}_{}'.format(save_path_suffix, args.obs_emb)

    bpl = model.BPL(lib_dir=lib_dir)
    bpl.load_inference_network('save/bpl_inference_network{}'.format(
        save_path_suffix))
    # omniglot_test_dataset = data.OmniglotDataset(data.train_img_dir,
    #                                              data.train_motor_dir)

    # num_test_images = 9
    # test_images = omniglot_test_dataset[0][:num_test_images]
    test_images = torch.load('test_images.pt')

    # Plotting
    fig, axss = plt.subplots(3, 6, figsize=(6 * 2, 3 * 2))

    for test_image, (i, j) in zip(test_images, get_test_image_ij()):
        axss[i, j].imshow(1 - test_image, cmap='gray')

    for test_image, (i, j) in zip(test_images, get_parse_ij()):
        try:
            posterior = bpl.posterior_results(
                num_traces=args.num_particles,
                inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK,
                observe={'image': test_image}
            )
            char_type_mode, char_token_mode = posterior.mode
            plot_parse(axss[i, j], char_token_mode)
            print('Effective sample size = {}'.format(
                posterior.effective_sample_size.item()))
        except:
            print('some error')

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

    filename = 'plots/parse{}.pdf'.format(save_path_suffix)
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--obs-emb', default='cnn2d5c',
                        help='cnn2d5c or alexnet')
    parser.add_argument('--small-lib', action='store_true',
                        help='use 250 primitives')
    parser.add_argument('--num-particles', type=int, default=10,
                        help=' ')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()
    main(args)
