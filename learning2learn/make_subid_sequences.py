import pickle
import scipy.io as sio
from pybpl.fit_hyperparameters.dataset import PreprocessedDataset


def main():
    D = sio.loadmat('./data_background_splines.mat')
    dataset = PreprocessedDataset(
        splines=D['bspline_substks'],
        drawings=D['pdrawings_norm'],
        scales=D['pdrawings_scales']
    )
    sequences = dataset.make_subid_dataset()
    with open('./subid_sequences_background.p', 'wb') as fp:
        pickle.dump(sequences, fp, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()