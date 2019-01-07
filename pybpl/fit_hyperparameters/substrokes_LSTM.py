"""
Train an LSTM to predict the next sub-stroke given the previous ones
"""

from __future__ import division, print_function
import os
import argparse
try:
    import pickle # python 3.x
except ImportError:
    import cPickle as pickle # python 2.x
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Masking, Embedding, TimeDistributed, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='subid_sequences.p', type=str)
parser.add_argument('--save_file', default='lstm.h5', type=str)
parser.add_argument('--max_len', default=None, type=int)
parser.add_argument('--nb_epoch', default=100, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--embedding_dim', default=50, type=int)
parser.add_argument('--lstm_dim', default=50, type=int)
parser.add_argument('--no-embedding', dest='embedding', action='store_false')
parser.set_defaults(embedding=True)
args = parser.parse_args()


def to_categorical(X, num_classes, mask_value=0):
    n, max_len = X.shape
    X1 = np.zeros((n, max_len, num_classes), dtype=np.float32)
    for i in range(n):
        for j in range(max_len):
            if X[i, j] != mask_value:
                X1[i, j, X[i,j]] = 1.

    return X1

def load_sequences(data_dir):
    """
    Load the data set of sequences.

    Parameters
    ----------
    data_dir : str
        path to data file

    Returns
    -------
    seqs : list of lists
        List of sequence samples. Each sequence samples is a list of ints
        within the range [1, vocab_size] inclusive. Samples may have varying
        length
    vocab_size : int
        size of the token vocabulary
    """
    vocab_size = 1212
    with open(data_dir, 'rb') as fp:
        seqs = pickle.load(fp)
    seqs1 = []
    for seq in seqs:
        seq1 = []
        for elt in seq:
            seq1.append(elt + 1)
        seqs1.append(seq1)

    return seqs1, vocab_size

def get_inputs(seqs, vocab_size, max_len):
    """
    Build the input array from a list of variable-length sequences. Add new
    'start' token at beginning of each sequence. Truncate sequences longer than
    max_len, and pad sequences shorter than max_len.

    Parameters
    ----------
    seqs : list of lists
        sequence samples. Each sequence samples is a list of ints within the
        range [1, vocab_size] inclusive. Samples may have varying length
    vocab_size : int
        size of the token vocabulary
    max_len : int
        maximum sequence length; longer sequences will be truncated

    Returns
    -------
    X : (n,m) ndarray
        data set; contains n sequences, each of length m

    """
    # if max_len is not specified, set it to equal the maximum sequence length
    if max_len is None:
        max_len = max([len(s) for s in seqs])
    # add 'start' token to beginning of each sequence
    seqs = [[vocab_size+1]+s for s in seqs]
    # create the data matrix
    X = pad_sequences(seqs, max_len, padding='post', truncating='post')
    if not args.embedding:
        X = to_categorical(X, num_classes=vocab_size+2)

    return X

def get_targets(seqs, vocab_size, max_len):
    """
    Build the target array from a list of variable-length sequences. Add new
    'end' token at end of each sequence. Truncate sequences longer than
    max_len, and pad sequences shorter than max_len.

    Parameters
    ----------
    seqs : list of lists
        sequence samples. Each sequence samples is a list of ints within the
        range [1, vocab_size] inclusive. Samples may have varying length
    vocab_size : int
        size of the token vocabulary
    max_len : int
        maximum sequence length; longer sequences will be truncated

    Returns
    -------
    Y : (n,m,v) ndarray
        shifted data set
    """
    # if max_len is not specified, set it to equal the maximum sequence length
    if max_len is None:
        max_len = max([len(s) for s in seqs])
    # add 'end' token to end of each sequence
    seqs = [s+[vocab_size+1] for s in seqs]
    # create the data matrix
    Y = pad_sequences(seqs, max_len, padding='post', truncating='post')
    # one-hot-encode the categories
    Y = utils.to_categorical(Y, num_classes=vocab_size+2)

    return Y

def build_model(vocab_size, embedding_dim, lstm_dim, dropout=0.3):
    """
    Build the LSTM model for next-token prediction

    Parameters
    ----------
    vocab_size : int
        size of the vocabulary of the input data
    embedding_dim : int
        dimensionality of the token embeddings
    lstm_dim : int
        dimensionality of the lstm layer
    dropout : float
        dropout probability

    Returns
    -------
    model : Sequential
        compiled Keras model
    """
    model = Sequential()
    if args.embedding:
        model.add(Embedding(vocab_size+2, embedding_dim, mask_zero=True))
    else:
        model.add(Masking(mask_value=0., input_shape=(args.max_len+1, vocab_size+2)))
    model.add(Dropout(dropout))
    model.add(LSTM(lstm_dim, dropout=dropout, recurrent_dropout=0.1, return_sequences=True))
    model.add(TimeDistributed(Dense(vocab_size+2, activation='softmax')))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )

    return model

def main():
    #config = tf.ConfigProto(device_count={'GPU':0})
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0')
    )
    sess = tf.Session(config=config)
    K.set_session(sess)

    seqs, vocab_size = load_sequences(args.data_dir)
    X = get_inputs(seqs, vocab_size, args.max_len)
    Y = get_targets(seqs, vocab_size, args.max_len)
    print('X shape: ', X.shape)
    print('Y shape: ', Y.shape)

    model = build_model(vocab_size, args.embedding_dim, args.lstm_dim)
    if os.path.isfile(args.save_file):
        os.remove(args.save_file)
    checkpoint = ModelCheckpoint(
        args.save_file,
        monitor='val_loss',
        save_best_only=True
    )
    model.fit(
        X, Y, epochs=args.nb_epoch, batch_size=args.batch_size,
        verbose=1, validation_split=0.25, shuffle=True,
        callbacks=[checkpoint]
    )

if __name__ == "__main__":
    main()
