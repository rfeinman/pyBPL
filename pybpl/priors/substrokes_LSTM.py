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
    """
    return

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

def build_model(vocab_size):
    """
    Build the LSTM model for next-token prediction

    Parameters
    ----------
    vocab_size : int
        size of the vocabulary of the input data

    Returns
    -------
    model : Sequential
        compiled Keras model
    """
    model = Sequential([
        Embedding(input_dim=vocab_size+1, output_dim=50, mask_zero=True),
        LSTM(50, return_sequences=True),
        TimeDistributed(Dense(vocab_size+1, activation='softmax'))
    ])
    model.compile(
        loss='categorical_crossentropy', optimizer='rmsprop',
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    args = parser.parse_args()

    X, vocab_size = load_data(args.data_dir)
    Y = shift_tokens(X)

    model = build_model(vocab_size)
    model.fit(
        X, Y, nb_epoch=args.nb_epoch, batch_size=args.batch_size,
        verbose=1, validation_split=0.25, shuffle=True
    )