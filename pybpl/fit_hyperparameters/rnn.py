from tqdm import tqdm
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


def preprocess_sequences(seqs, vocab_size, max_len, one_hot_targets=True):
    # add 'start' token to head of each input sequence
    seqs_input = [[vocab_size]+s[:-1] for s in seqs]
    # add 1 to each input subID since '0' is saved for padding
    seqs_input = [np.array(s)+1 for s in seqs_input]
    # create the input and target arrays.
    X = pad_sequences(seqs_input, max_len, padding='post', truncating='post')
    Y = pad_sequences(seqs, max_len, padding='post', truncating='post')
    if one_hot_targets:
        Y = to_categorical(Y, num_classes=vocab_size)

    return X, Y

def predict(model, X, sess, batch_size=128):
    # build TF graph
    x = tf.placeholder(dtype=tf.int32, shape=(None,)+X.shape[1:])
    y_pred = model(x)

    # predict
    Y_pred = []
    nb_batches = int(np.ceil(X.shape[0] / batch_size))
    with sess.as_default():
        for i in tqdm(range(nb_batches)):
            y_pred_val = sess.run(
                y_pred,
                feed_dict={x: X[i*batch_size:(i+1)*batch_size],
                           K.learning_phase():0}
            )
            Y_pred.append(y_pred_val)
    Y_pred = np.concatenate(Y_pred)

    return Y_pred