import model
import pyprob.diagnostics


def main(args):
    if args.cuda:
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

    bpl = model.BPL(lib_dir=lib_dir)
    bpl.load_inference_network('save/bpl_inference_network{}'.format(
        save_path_suffix))

    pyprob.diagnostics.network(bpl._inference_network,
                               'plots/inf_net{}'.format(save_path_suffix))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--obs-emb', default='cnn2d5c',
                        help='cnn2d5c or alexnet')
    parser.add_argument('--large-lib', action='store_true',
                        help='use 1250 primitives')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()
    main(args)
