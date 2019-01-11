import pickle
import torch
import numpy as np
from keras.models import load_model

from pybpl.library import Library
from pybpl.model.type_dist import StrokeTypeDist
from train_rnn_subids import preprocess_sequences

VOCAB_SIZE = 1212 # number of unique primitive IDs


def score_sequence_HMM(sequence, type_dist, max_len):
    sequence = torch.tensor(sequence, dtype=torch.long)
    log_probs = type_dist.score_subIDs(sequence)
    score = log_probs[:max_len].sum()

    return score.item()

def score_sequence_RNN(sequence, y_pred):
    log_prob = 0.
    timesteps = min(len(sequence), 10)
    for i in range(timesteps):
        # add 1 to all IDs since we used '0' for padding
        lp = np.log(y_pred[i, sequence[i]+1])
        log_prob = log_prob + lp

    return log_prob

def main():
    print('Loading evaluation sequences...')
    with open('./subid_sequences_evaluation.p', 'rb') as fp:
        sequences = pickle.load(fp)
    print('Number of sequences to score: %i\n' % len(sequences))

    # compute HMM scores
    print('Getting HMM scores...')
    lib = Library()
    type_dist = StrokeTypeDist(lib)
    scoresHMM = [score_sequence_HMM(seq, type_dist, 10) for seq in sequences]
    print('mean HMM score: %0.4f' % np.mean(scoresHMM))
    print('std: %0.4f\n' % np.std(scoresHMM))

    # compute RNN scores
    print('Getting RNN scores...')
    X, Y = preprocess_sequences(sequences, VOCAB_SIZE, 10)
    model = load_model('./rnn_subids.h5')
    Y_pred = model.predict(X, verbose=1)
    scoresRNN = [score_sequence_RNN(sequences[i], Y_pred[i])
                 for i in range(len(sequences))]
    print('mean RNN score: %0.4f' % np.mean(scoresRNN))
    print('std: %0.4f\n' % np.std(scoresRNN))

if __name__ == '__main__':
    main()