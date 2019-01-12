"""

Train an LSTM to predict the next sub-stroke ID
given the previous ones.

No word embeddings are used right now.

"""
from __future__ import division, print_function
try:
    import pickle # python 3.x
except ImportError:
    import cPickle as pickle # python 2.x
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Embedding
from keras.layers import SpatialDropout1D, GRU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam


parser = argparse.ArgumentParser()
parser.add_argument('--data_file', default='./subid_sequences_background.p', type=str)
parser.add_argument('--save_file', default='./rnn_subids.h5', type=str)
parser.add_argument('--max_len', default=10, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--nb_epochs', default=30, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--gpu', default=False, action='store_true')
ARGS = parser.parse_args()
VOCAB_SIZE = 1212 # 1212 primitive IDs



def preprocess_sequences(sequences, vocab_size, max_len):
    N = len(sequences)
    # add 'start' token to head of each sequence
    seqs_input = [[vocab_size]+s[:-1] for s in sequences]
    # create the input and target arrays.
    X = np.zeros((N, max_len), dtype=np.int16)
    Y = np.zeros((N, max_len, vocab_size), dtype=np.float32)
    for i in range(N):
        timesteps = min(max_len, len(sequences[i]))
        seq_input = np.asarray(seqs_input[i], dtype=np.int16) + 1
        seq_target = np.asarray(sequences[i], dtype=np.int16)
        for t in range(timesteps):
            X[i,t] = seq_input[t]
            Y[i,t,seq_target[t]] = 1.

    return X, Y

def build_model(vocab_size, dropout):
    """
    Build the LSTM model for sequence prediction. This model is trained
    to predict the next subid given the previous ones.
    """
    model = Sequential()
    model.add(Embedding(vocab_size+2, 128, mask_zero=True))
    model.add(SpatialDropout1D(dropout))
    model.add(GRU(128, dropout=dropout, recurrent_dropout=dropout,
                   return_sequences=True))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(clipnorm=1.),
        metrics=['accuracy']
    )

    return model

def main():
    # set TF session
    print('Initializing TF session...')
    if ARGS.gpu:
        gpu_count = 1
    else:
        gpu_count = 0
    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU':gpu_count}))
    K.set_session(sess)

    # load sequences. This is a list of lists
    print('Loading subid sequences...')
    with open(ARGS.data_file, 'rb') as fp:
        sequences = pickle.load(fp)
    print('Example sequences:')
    for i in range(3):
        print('\t', sequences[i])

    # get input and target arrays
    print('Building training data arrays...')
    X, Y = preprocess_sequences(sequences, VOCAB_SIZE, ARGS.max_len)
    print('X shape: ', X.shape)
    print('Y shape: ', Y.shape)

    # initialize the network
    print('Initializing neural network...')
    model = build_model(VOCAB_SIZE, ARGS.dropout)
    checkpoint = ModelCheckpoint(
        ARGS.save_file,
        monitor='val_loss',
        save_best_only=True
    )

    # train the network
    print('Training the network...')
    hist = model.fit(
        X, Y, epochs=ARGS.nb_epochs, batch_size=ARGS.batch_size,
        validation_split=0.2, shuffle=True,
        callbacks=[checkpoint]
    )

    # report best results
    best_ix = np.argmin(hist.history['val_loss'])
    train_loss = hist.history['loss'][best_ix]
    valid_loss = hist.history['val_loss'][best_ix]
    print('Best result - train_loss: %0.4f   valid_loss: %0.4f' %
          (train_loss, valid_loss))

if __name__ == "__main__":
    main()



