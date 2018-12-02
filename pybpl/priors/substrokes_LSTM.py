"""
Train an LSTM to predict the next sub-stroke given the previous ones
"""

from __future__ import division, print_function
import argparse
from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--nb_epoch', default=100, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--embedding_dim', default=50, type=int)
parser.add_argument('--lstm_dim', default=50, type=int)

def load_data(data_dir):
    """
    Load the data set

    Parameters
    ----------
    data_dir : str
        path to data file

    Returns
    -------
    X : (n,m) ndarray
        data set; contains n sequences, each of length m
    vocab_size : int
        size of the token vocabulary
    """
    X = None
    vocab_size = 1000

    return X, vocab_size

def shift_tokens(X):
    """
    Shift all tokens ahead by one time-step, thereby creating the prediction
    targets

    Parameters
    ----------
    X : (n,m) ndarray
        data set

    Returns
    -------
    Y : (n,m) ndarray
        shifted data set
    """
    return

def build_model(vocab_size, embedding_dim, lstm_dim):
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

    Returns
    -------
    model : Sequential
        compiled Keras model
    """
    model = Sequential([
        Embedding(vocab_size+1, embedding_dim, mask_zero=True),
        LSTM(lstm_dim, return_sequences=True),
        TimeDistributed(Dense(vocab_size+1, activation='softmax'))
    ])
    model.compile(
        loss='categorical_crossentropy', optimizer='rmsprop',
        metrics=['accuracy']
    )

    return model

def main():
    args = parser.parse_args()

    X, vocab_size = load_data(args.data_dir)
    Y = shift_tokens(X)

    model = build_model(vocab_size, args.embedding_dim, args.lstm_dim)
    model.fit(
        X, Y, nb_epoch=args.nb_epoch, batch_size=args.batch_size,
        verbose=1, validation_split=0.25, shuffle=True
    )

if __name__ == "__main__":
    main()