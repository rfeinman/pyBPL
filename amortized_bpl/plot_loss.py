import matplotlib.pyplot as plt
import model
import pyprob.diagnostics


def main(args):
    if args.small_lib:
        lib_dir = '../lib_data250'
        save_path_suffix = '_250'
    else:
        lib_dir = '../lib_data'
        save_path_suffix = ''

    if args.obs_emb == 'cnn2d5c':
        pass
    else:
        save_path_suffix = '{}_{}'.format(save_path_suffix, args.obs_emb)

    bpl = model.BPL(lib_dir=lib_dir)
    bpl.load_inference_network('save/bpl_inference_network{}'.format(
        save_path_suffix))

    trace_per_second = bpl._inference_network._total_train_traces / \
        bpl._inference_network._total_train_seconds
    print('Average traces/second: {:.2f}'.format(trace_per_second))
    fig, ax = plt.subplots(1, 1, dpi=200)
    ax.plot(bpl._inference_network._history_train_loss_trace,
            bpl._inference_network._history_train_loss)
    ax.set_xlabel('traces')
    ax.set_ylabel('loss')

    fig.tight_layout(pad=0)

    filename = 'plots/loss{}.pdf'.format(save_path_suffix)
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))

    # pyprob.diagnostics.network(bpl._inference_network, 'plots/inf_net{}'.format(save_path_suffix))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--obs-emb', default='cnn2d5c',
                        help='cnn2d5c or alexnet')
    parser.add_argument('--small-lib', action='store_true',
                        help='use 250 primitives')
    args = parser.parse_args()
    main(args)
