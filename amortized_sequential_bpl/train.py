import os
import model
import pyprob
import signal
import sys
import pybpl


def save_bpl_inference_network(bpl, save_path_suffix):
    dir_ = 'save'
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    path = os.path.join(dir_, 'bpl_inference_network{}'.format(
        save_path_suffix))
    bpl.save_inference_network(path)
    print('Saved to {}'.format(path))


def train(args):
    print(args)

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

    if args.obs_emb == 'cnn2d5c':
        embedding = pyprob.ObserveEmbedding.CNN2D5C
    else:
        embedding = pyprob.ObserveEmbedding.ALEXNET
    save_path_suffix = '{}_{}'.format(save_path_suffix, args.obs_emb)

    if args.only_categoricals:
        pybpl.set_train_non_categoricals(False)
        save_path_suffix = '{}_cat'.format(save_path_suffix)

    bpl = model.BPL(lib_dir=lib_dir)
    if args.train_from_scratch:
        print('training from scratch')
    else:
        try:
            bpl.load_inference_network('save/bpl_inference_network{}'.format(
                save_path_suffix))
            print('continuing to train')
        except RuntimeError:
            print('training from scratch')

    def signal_handler(sig, frame):
        print('')
        print('signal.SIGINT = {}'.format(sig))
        print('You pressed Ctrl+C!')
        save_bpl_inference_network(bpl, save_path_suffix)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    total_num_traces = 0
    while total_num_traces < args.num_traces:
        num_traces = min(args.save_every,
                         args.num_traces - total_num_traces)
        bpl.learn_inference_network(
            num_traces=num_traces,
            batch_size=args.batch_size,
            observe_embeddings={'image': {
                'dim': 128,
                'embedding': embedding,
                'reshape': (1, 105, 105)}},
            inference_network=pyprob.InferenceNetwork.LSTM)
        save_bpl_inference_network(bpl, save_path_suffix)
        total_num_traces += num_traces


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=1,
                        help=' ')
    parser.add_argument('--obs-emb', default='cnn2d5c',
                        help='cnn2d5c or alexnet')
    parser.add_argument('--num-traces', type=int, default=10,
                        help=' ')
    parser.add_argument('--save-every', type=int, default=2,
                        help=' ')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--train-from-scratch', action='store_true')
    parser.add_argument('--only-categoricals', action='store_true',
                        help='train proposals for categoricals only')
    parser.add_argument('--large-lib', action='store_true',
                        help='use 1250 primitives')
    args = parser.parse_args()
    train(args)
