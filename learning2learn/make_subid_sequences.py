try:
    import pickle # python 3.x
except ImportError:
    import cPickle as pickle # python 2.x
import argparse
import scipy.io as sio
from pybpl.fit_hyperparameters.dataset import PreprocessedDataset

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)
ARGS = parser.parse_args()


def main():
    if ARGS.mode == 'train':
        data_file = './data_background_splines.mat'
        save_file = './subid_sequences_background.p'
    elif ARGS.mode == 'test':
        data_file = './data_evaluation_splines.mat'
        save_file = './subid_sequences_evaluation.p'
    else:
        raise Exception("'mode' argument must be either 'train' or 'test'")
    D = sio.loadmat(data_file)
    dataset = PreprocessedDataset(
        splines=D['bspline_substks'],
        drawings=D['pdrawings_norm'],
        scales=D['pdrawings_scales']
    )
    sequences = dataset.make_subid_dataset()
    with open(save_file, 'wb') as fp:
        pickle.dump(sequences, fp, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()