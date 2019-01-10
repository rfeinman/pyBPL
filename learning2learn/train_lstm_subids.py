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
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Embedding
from keras.layers import LSTM, GRU, SpatialDropout1D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop, Adam

DATA_FILE = './subid_sequences.p'
MAX_LEN = 10 # truncate sequences longer than 10
VOCAB_SIZE = 1212 # 1212 primitive IDs
EPOCHS = 30
BATCH_SIZE = 64

config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list='0'))
#config = tf.ConfigProto(device_count={'GPU':0})
sess = tf.Session(config=config)
K.set_session(sess)


def preprocess_sequences(sequences, vocab_size, max_len):
    N = len(sequences)
    seqs_input = [[vocab_size]+s[:-1] for s in sequences]
    # create the input and target arrays.
    X = np.zeros((N, max_len), dtype=np.int16)
    Y = np.zeros((N, max_len, vocab_size+1), dtype=np.float32)
    for i in range(N):
        timesteps = min(max_len, len(seqs_input[i]))
        seq = np.asarray(sequences[i], dtype=np.int16) + 1
        # add 'start' token to head of each sequence
        X[i,0] = vocab_size+1
        for t in range(timesteps):
            if t < max_len-2:
                X[i,t+1] = seq[t]
            Y[i,t,seq[t]] = 1.

    return X, Y

def build_model(vocab_size, dropout=0.5):
    """
    Build the LSTM model for sequence prediction. This model is trained
    to predict the next subid given the previous ones.
    """
    model = Sequential()
    model.add(Embedding(vocab_size+2, 128, mask_zero=True))
    model.add(SpatialDropout1D(dropout))
    model.add(LSTM(128, dropout=dropout, recurrent_dropout=dropout,
                   return_sequences=True))
    model.add(TimeDistributed(Dense(vocab_size+1, activation='softmax')))
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(clipnorm=1.),
        metrics=['accuracy']
    )

    return model

def main():
    # load sequences. This is a list of lists
    print('Loading subid sequences...')
    with open(DATA_FILE, 'rb') as fp:
        sequences = pickle.load(fp)
    print('Example sequences:')
    for i in range(3):
        print('\t', sequences[i])

    print('Building training data arrays...')
    # get input and target arrays
    X, Y = preprocess_sequences(sequences, VOCAB_SIZE, MAX_LEN)
    print('X shape: ', X.shape)
    print('Y shape: ', Y.shape)

    print('Initializing neural network...')
    model = build_model(VOCAB_SIZE)
    checkpoint = ModelCheckpoint(
        'lstm_subids.h5',
        monitor='val_loss',
        save_best_only=True
    )
    print('Training the network...')
    hist = model.fit(
        X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.2, shuffle=True,
        callbacks=[checkpoint]
    )
    best_loss = np.min(hist.history['val_loss'])
    print('best validation loss: %0.4f' % best_loss)

if __name__ == "__main__":
    main()



