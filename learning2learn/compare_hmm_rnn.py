import pickle
import torch
import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.models import load_model

from pybpl.library import Library
from pybpl.model.type_dist import StrokeTypeDist
from pybpl.fit_hyperparameters.rnn import preprocess_sequences, predict

MAX_LEN = 10
VOCAB_SIZE = 1212 # number of unique primitive IDs



def score_sequence_HMM(sequence, type_dist, max_len):
    sequence = torch.tensor(sequence, dtype=torch.long)
    log_probs = type_dist.score_subIDs(sequence)
    score = log_probs[:max_len].sum()

    return score.item()

def score_sequence_RNN(sequence, y_pred, max_len):
    log_prob = 0.
    timesteps = min(len(sequence), max_len)
    for i in range(timesteps):
        lp = np.log(y_pred[i, sequence[i]])
        log_prob = log_prob + lp

    return log_prob

def main():
    # initalize TF session
    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    K.set_session(sess)

    print('Loading evaluation sequences...')
    with open('./subid_sequences_evaluation.p', 'rb') as fp:
        sequences = pickle.load(fp)
    print('Number of sequences to score: %i\n' % len(sequences))

    # compute HMM scores
    print('Getting HMM scores...')
    lib = Library()
    type_dist = StrokeTypeDist(lib)
    scoresHMM = [score_sequence_HMM(seq, type_dist, MAX_LEN) for seq in sequences]
    print('mean HMM score: %0.4f' % np.mean(scoresHMM))
    print('std: %0.4f\n' % np.std(scoresHMM))

    # compute RNN scores
    print('Getting RNN scores...')
    X, _ = preprocess_sequences(sequences, VOCAB_SIZE, MAX_LEN)
    print(X.shape)
    model = load_model('./rnn_subids.h5')
    Y_pred = predict(sess, model, X)
    scoresRNN = [score_sequence_RNN(sequences[i], Y_pred[i], MAX_LEN)
                 for i in range(len(sequences))]
    print('mean RNN score: %0.4f' % np.mean(scoresRNN))
    print('std: %0.4f\n' % np.std(scoresRNN))

if __name__ == '__main__':
    main()