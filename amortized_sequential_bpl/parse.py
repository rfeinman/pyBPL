import model
import pyprob
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import pybpl
import data
import numpy as np


def get_color(k):
    """Color map for the stroke of index k"""

    scol = ['r', 'g', 'b', 'm', 'c']
    ncol = len(scol)
    if k < ncol:
        out = scol[k]
    else:
        out = scol[-1]
    return out


def plot_parse(ax, character_token, **kwargs):
    for i, stroke_token in enumerate(character_token.stroke_tokens):
        color = get_color(i)
        for j, xy in enumerate(stroke_token.motor):
            if j == 0:
                ax.scatter(xy[0, 0], -xy[0, 1], s=20, color=color, marker='o', **kwargs)
            ax.plot(xy[:, 0], -xy[:, 1], color=color, linewidth=2, **kwargs)
    return ax


def get_test_image_ij():
    for i in range(3):
        for j in range(3):
            yield i, j


def get_parse_ij():
    for i in range(3):
        for j in range(3):
            yield i, j + 3


def get_image_from_prior(model):
    trace = model.get_trace()
    image = trace.variables_observed[0].value
    return image


def main(args):
    if args.cuda:
        pyprob.set_device('cuda')
        save_path_suffix = '_cuda'
    else:
        save_path_suffix = ''

    if args.large_lib:
        lib_dir = '../lib_data'
        save_path_suffix = '{}'.format(save_path_suffix)
    else:
        lib_dir = '../lib_data250'
        save_path_suffix = '{}_250'.format(save_path_suffix)

    save_path_suffix = '{}_{}'.format(save_path_suffix, args.obs_emb)

    if args.only_categoricals:
        pybpl.set_train_non_categoricals(False)
        save_path_suffix = '{}_cat'.format(save_path_suffix)

    bpl = model.BPL(lib_dir=lib_dir)
    bpl.load_inference_network('save/bpl_inference_network{}'.format(
        save_path_suffix))

    num_test_images = 9
    if args.synthetic:
        test_images = [get_image_from_prior(bpl) for _ in range(num_test_images)]
        synthetic_suffix = '_synthetic'
    else:
        if args.random_real:
            omniglot_test_dataset = data.OmniglotDataset(data.train_img_dir,
                                                         data.train_motor_dir)
            test_images = [omniglot_test_dataset[np.random.choice(len(omniglot_test_dataset))][0] for _ in range(num_test_images)]
            synthetic_suffix = '_random_real'
        else:
            # omniglot_test_dataset = data.OmniglotDataset(data.train_img_dir,
            #                                              data.train_motor_dir)
            # test_images = omniglot_test_dataset[0][:num_test_images]
            test_images = torch.load('test_images.pt')
            synthetic_suffix = ''

    # Plotting
    fig, axss = plt.subplots(3, 6, figsize=(6 * 2, 3 * 2))
    big_fig, big_axss = plt.subplots(num_test_images, 1 + args.num_particles,
                                     figsize=((1 + args.num_particles) * 2,
                                              num_test_images * 2))

    for idx, (test_image, (i, j)) in enumerate(zip(test_images, get_test_image_ij())):
        axss[i, j].imshow(1 - test_image, cmap='gray')
        big_axss[idx, 0].imshow(1 - test_image, cmap='gray')

    inference_engine = \
        pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK
    for test_image_idx, (test_image, (i, j)) in enumerate(zip(test_images, get_parse_ij())):
        try:
            posterior = bpl.posterior_results(
                num_traces=args.num_particles,
                inference_engine=inference_engine,
                observe={'image': test_image}
            )
            character_token_mode = posterior.mode
            plot_parse(axss[i, j], character_token_mode)
            print('Effective sample size = {}'.format(
                posterior.effective_sample_size.item()))

            idx_descending_log_weights = sorted(range(args.num_particles), key=lambda x: posterior._log_weights[x], reverse=True)
            for i in range(args.num_particles):
                ax = big_axss[test_image_idx, 1 + i]
                character_token = posterior.get_values()[idx_descending_log_weights[i]]
                normalized_weight = posterior.weights_numpy()[idx_descending_log_weights[i]]
                log_weight = posterior._log_weights[idx_descending_log_weights[i]]
                plot_parse(ax, character_token)
                ax.text(0.9, 0.9, '{:.2f} ({:.2f})'.format(normalized_weight, log_weight),
                        horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

            # for i, (char_type, character_token) in enumerate(posterior.get_values()):
            #     ax = big_axss[test_image_idx, 1 + i]
            #     plot_parse(ax, character_token)
            #     ax.text(0.9, 0.9, '{:.2f} ({:.2f})'.format(posterior.weights_numpy()[i], posterior._log_weights[i]),
            #             horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
        except Exception as e:
            print(e)
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

    filename = 'plots/parse{}{}.pdf'.format(save_path_suffix, synthetic_suffix)
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


    for axs in big_axss:
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim(105, 0)
            ax.set_xlim(0, 105)
    big_fig.legend([Line2D([0], [0], color=get_color(i), lw=2)
                for i in range(5)],
               range(1, 6),
               ncol=5, loc=(0.6, -0.01), frameon=False)
    big_fig.tight_layout(pad=0)
    filename = 'plots/parse_big{}{}.pdf'.format(save_path_suffix, synthetic_suffix)
    big_fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--obs-emb', default='cnn2d5c',
                        help='cnn2d5c or alexnet')
    parser.add_argument('--large-lib', action='store_true',
                        help='use 1250 primitives')
    parser.add_argument('--num-particles', type=int, default=10,
                        help=' ')
    parser.add_argument('--only-categoricals', action='store_true',
                        help='train proposals for categoricals only')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--synthetic', action='store_true',
                        help='condition on images from prior')
    parser.add_argument('--random-real', action='store_true',
                        help='random real data')
    args = parser.parse_args()
    main(args)
