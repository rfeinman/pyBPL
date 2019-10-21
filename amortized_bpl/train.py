import os
import model
import pyprob
import signal
import sys


def save_bpl_inference_network(bpl):
    dir_ = 'save'
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    path = os.path.join(dir_, 'bpl_inference_network')
    bpl.save_inference_network(path)
    print('Saved to {}'.format(path))


def train(args):
    print(args)

    if args.cuda:
        pyprob.set_device('cuda')
    bpl = model.BPL()

    def signal_handler(sig, frame):
        print('')
        print('signal.SIGINT = {}'.format(sig))
        print('You pressed Ctrl+C!')
        save_bpl_inference_network(bpl)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    bpl.learn_inference_network(
        num_traces=args.num_traces,
        batch_size=args.batch_size,
        observe_embeddings={'image': {
            'dim': 32,
            'embedding': pyprob.ObserveEmbedding.CNN2D5C,
            'reshape': (1, 105, 105)}},
        inference_network=pyprob.InferenceNetwork.LSTM)

    save_bpl_inference_network(bpl)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=1,
                        help=' ')
    parser.add_argument('--num-traces', type=int, default=10,
                        help=' ')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()
    train(args)
